#include "flowfield.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "lib/framework/debug.h"
#include "lib/framework/vector.h"
#include "lib/ivis_opengl/pieblitfunc.h"
#include "lib/ivis_opengl/piepalette.h"
#include "lib/ivis_opengl/textdraw.h"

#include "display3d.h"
#include "map.h"
#include "raycast.h"

// FIXME: redefinition of "debug" from "debug.h" to avoid conflict with Qt, which has "debug" function in qlogging.h
//   As Qt is being phased-out from Warzone, this is only temporary
//   We should really move to C++ instead of using macros :(
#undef debug

#include <qthreadpool.h>
#include <qelapsedtimer.h>

// FIXME: see above
/**
 * Output printf style format str with additional arguments.
 * Only outputs if debugging of part was formerly enabled with debug_enable_switch.
 */
#define wz_debug(part, ...) do { if (enabled_debug[part]) _debug(__LINE__, part, __FUNCTION__, __VA_ARGS__); } while(0)
#define debug(part, ...) do { if (enabled_debug[part]) _debug(__LINE__, part, __FUNCTION__, __VA_ARGS__); } while(0)

// C++11 does not have std::make_unique by a mistake. Let's fix that omission.
namespace std {
    template<typename T, typename... Args>
    unique_ptr<T> make_unique(Args &&... args) {
        return unique_ptr<T>(new T(forward<Args>(args)...));
    }
}

namespace flowfield {
    namespace detail {
        // Sector is a square with side length of SECTOR_SIZE.
        constexpr const unsigned int SECTOR_SIZE = 16;

        constexpr const unsigned short NOT_PASSABLE = std::numeric_limits<unsigned short>::max();
        constexpr const unsigned short COST_MIN = 1;

        // Decides how much slopes should be avoided
        constexpr const float SLOPE_COST_BASE = 0.05f;
        // Decides when terrain height delta is considered a slope
        // TODO: I do not have much knowledge about WZ, but why almost every tile has different heights?
        constexpr const unsigned short SLOPE_THRESOLD = 10;

        // If an exception is thrown in thread pool FuturedTask, should we terminate or try to continue?
        constexpr const bool EXCEPTION_IN_THREADPOOL_SHOULD_TERMINATE = true;

        /**
         * "-1" is never expire. Otherwise idle threads are expired after X milliseconds.
         * It's better to keep the threads available all the time, so there is no startup delay.
         */
        constexpr const int THREAD_POOL_EXPIRY_TIMEOUT_MS = -1;

        /*
        * How many portal-level A* paths to cache (LRU)
        * Notice: this is per type of propulsion
        */
        constexpr const unsigned short PORTAL_PATH_CACHE_MAX = 50;

        /*
        * How many sector-level entries to cache (LRU)
        * Total number of unique sector-flowfields depends directly on (number_of_portals ^ 2)
        * Each flowfield takes SECTOR_SIZE^2 * sizeof(VectorT)
        * For example, max cache size for defaults could be 64 * 2*4 * 8096 = 4MB
        * Notice: this is per type of propulsion
        */
        constexpr const unsigned short FLOWFIELD_CACHE_MAX = 8096;

        /*
        * How many parallel path finding requests can be processed at once.
        * Note that after finding portal-level path, the main task waits for it subtasks to complete, which use default pool.
        * Parallelism greater than numer of logical cores might give spikes of 100% CPU
        * but it might be not noticeable, because most of the time the task sleeps.
        */
        const int PORTAL_PATH_THREAD_POOL_MAX = QThread::idealThreadCount();

        // Thread pool used for portal-level A* pathfinding.
        // Those tasks wait for all subtasks to complete, so give them dedicated pool, to prevent deadlock
        QThreadPool portalPathThreadPool;

        struct RayCastCallbackData {
            unsigned int propulsionIndex;
            bool isPassable;
        };

        struct Tile {
            unsigned short cost;

            bool isBlocking() const;
        };

        class AbstractSector {
        public:
            typedef std::array<std::array<Tile, SECTOR_SIZE>, SECTOR_SIZE> tileArrayT;

            AbstractSector() = default;

            AbstractSector &operator=(AbstractSector &) = delete;

            AbstractSector &operator=(AbstractSector &&) = delete;

            AbstractSector(AbstractSector &) = delete;

            AbstractSector(AbstractSector &&) = default;

            virtual ~AbstractSector() = default;

            virtual void setTile(unsigned x, unsigned y, Tile tile) = 0;

            virtual Tile getTile(unsigned x, unsigned y) const = 0;

            virtual Tile getTile(Vector2i p) const = 0;

            virtual bool checkIsEmpty() const = 0; // Actual iterating through tiles
            virtual bool isEmpty() const; // If EmptySector or just Sector
            void addPortal(unsigned portalId);

            const std::vector<unsigned>& getPortals() const;

            static unsigned getIdByCoords(unsigned x, unsigned y, unsigned mapWidth);

            static unsigned getIdByCoords(unsigned x, unsigned y);

            static unsigned getIdByCoords(Vector2i p);

            static Vector2i getTopLeftCorner(unsigned id); // Top-left and bottom-right
            static Vector2i getTopLeftCornerByCoords(Vector2i point); // Top-left and bottom-right
            static std::vector<unsigned>
            getNeighbors(const std::vector<std::unique_ptr<AbstractSector>> &sectors, Vector2i center);

        protected:
            std::vector<unsigned> portalIds;
        };

        class Sector : public AbstractSector {
        public:
            using AbstractSector::AbstractSector;

            void setTile(unsigned x, unsigned y, Tile tile) override;

            Tile getTile(unsigned x, unsigned y) const override;

            Tile getTile(Vector2i p) const override;

            bool checkIsEmpty() const override;

        private:
            tileArrayT tiles{};
        };

        // Empty sector - optimization. Functions in this sector should always return COST_MIN.
        class EmptySector : public AbstractSector {
        public:
            using AbstractSector::AbstractSector;

            void setTile(unsigned x, unsigned y, Tile tile) override;

            Tile getTile(unsigned x, unsigned y) const override;

            Tile getTile(Vector2i p) const override;

            bool checkIsEmpty() const override;

            bool isEmpty() const override;
        };

        struct Portal {
            typedef std::vector<Vector2i> pointsT;

            // Sector layout
            // Right and bottom borders are "first sector points" in each sector
            // 1 - first sector points
            // 2 - second sector points
            /*   2 2 2 2|1
             *   2       1
             *   2       1
             *   2|1 1 1 1
            */

            AbstractSector *firstSector = nullptr;
            AbstractSector *secondSector = nullptr;
            pointsT firstSectorPoints;
            pointsT secondSectorPoints;
            std::vector<unsigned> neighbors;

            Portal() = default;

            Portal(const Portal&) = delete;

            Portal &operator=(const Portal&) = delete;

            Portal(Portal &&) = default;

            Portal &operator=(Portal &&) = default;

            ~Portal() = default;

            Portal(AbstractSector *sector1, AbstractSector *sector2, pointsT &firstSectorPoints,
                   pointsT &secondSectorPoints);

            bool isValid() const;

            Vector2i getFirstSectorCenter() const;

            Vector2i getSecondSectorCenter() const;
        };

        typedef std::vector<std::unique_ptr<AbstractSector>> sectorListT;
        // Map of [portalId, Portal]. Because portals can be removed, portal ids might be not continuous
        typedef std::map<unsigned, Portal> portalMapT;

        // Generic A* algorithm. Reimplemented to better suit needs of flowfield.
        // The one in astar.cpp is absolutely NOT reusable, unfortunately.
        // PLEASE KEEP THIS CODE HERE CLEAN!
        class AbstractAStar {
        public:
            AbstractAStar() = delete;

            AbstractAStar(const AbstractAStar&) = delete;

            AbstractAStar &operator=(const AbstractAStar&) = delete;

            AbstractAStar(AbstractAStar&&) = delete;

            AbstractAStar &operator=(AbstractAStar&&) = delete;

            explicit AbstractAStar(unsigned goal) : goal(goal) {};

            // Returns indexes of subsequent nodes in the path. Empty container means no path exists.
            std::deque<unsigned> findPath(unsigned startingIndex, unsigned nodes);

            virtual bool findPathExists(unsigned startingIndex, unsigned nodes);

        protected:
            // Implementation of following functions depends on data (tile, portal, ...)

            // Returns indexes of neighbors for given node
            virtual std::vector<unsigned> getNeighbors(unsigned index) = 0;

            // Returns distance between current node and considered node.
            // For grid, return "10 * tile.cost" for vertical or horizontal, and "14 * tile.cost" for diagonal. This avoids costly `sqrt(2)` for grids.
            virtual unsigned distance(unsigned current, unsigned neighbor) = 0;

            // The heuristic function. Returns expected cost of moving form `start` to `goal`. Use octal for diagonal and maybe Euclidean for portals
            // http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
            virtual unsigned heuristic(unsigned start) = 0;

            virtual ~AbstractAStar() = default;

            unsigned goal;

        private:
            struct Node {
                unsigned index;
                unsigned cost;
                unsigned heuristic;

                inline bool operator<(const Node &other) const {
                    // We want the top element to have lowest cost
                    return (cost + heuristic) > (other.cost + other.heuristic);
                }
            };

            // For each node, which node it can most efficiently be reached from
            std::map<unsigned, unsigned> cameFrom{};

            std::deque<unsigned> reconstructPath(unsigned start);

            unsigned _debugNodesVisited = 0;

            void logDebugNodesStats(unsigned nodesTotal, unsigned nodesInPath) const;
        };

        // This class works only in one given sector. `startingIndex` and `goal` therefore must be in range [0 ... SECTOR_SIZE^2)
        class TileAStar : public AbstractAStar {
        public:
            TileAStar(unsigned goal, const sectorListT &sectors);

            bool findPathExists(unsigned startingIndex, unsigned nodes) override;

        protected:
            std::vector<unsigned> getNeighbors(unsigned index) override;

            unsigned distance(unsigned current, unsigned neighbor) override;

            unsigned heuristic(unsigned start) override; // octile heuristic is preferred
        private:
            Vector2i goalPoint;
            const sectorListT &sectors;
            unsigned sectorId;

            static unsigned distanceCommon(Vector2i point1, Vector2i point2, unsigned cost);
        };

        class PortalAStar : public AbstractAStar {
        public:
            PortalAStar(unsigned goal, portalMapT &portals)
                    : AbstractAStar(goal), portals(portals), goalPortal(portals[goal]) {
            }

        protected:
            std::vector<unsigned> getNeighbors(unsigned index) override;

            unsigned distance(unsigned current, unsigned neighbor) override;

            unsigned heuristic(unsigned start) override; // straight-line (any angle) heuristic is preferred
        private:
            portalMapT &portals;
            const Portal &goalPortal;

            static unsigned distanceCommon(const Portal &portal1, const Portal &portal2);
        };

        // Mutex for logs and debug stuff
        std::mutex logMutex;

        template<typename T>
        class FuturedTask : public QRunnable {
        public:
            void run() final {
                try {
                    QElapsedTimer timer;
                    timer.start();

                    runPromised();

                    if (DEBUG_BUILD) // Mutex is costly and won't be optimized in release mode
                    {
                        std::lock_guard<std::mutex> lock(logMutex);
                        unsigned took = timer.elapsed();
                        wz_debug(LOG_FLOWFIELD, "FuturedTask (%s) took %d ms", typeid(*this).name(), took);
                    }
                } catch (const std::exception &ex) {
                    promise.set_exception(std::current_exception());
                    {
                        std::lock_guard<std::mutex> lock(logMutex);
                        wz_debug(LOG_ERROR, "Exception in thread pool worker: %s", ex.what());
                    }

                    if (EXCEPTION_IN_THREADPOOL_SHOULD_TERMINATE) {
                        std::terminate();
                    }
                }
            }

            std::future<T> getFuture() {
                return promise.get_future();
            }

            ~FuturedTask() override = default;

        protected:
            virtual void runPromised() = 0;

            void setPromise(T value) {
                promise.set_value(value);
            }

        private:
            std::promise<T> promise;
        };

        // Promise states that path request has been completed
        class PathRequestTask : public FuturedTask<bool> {
        public:
            PathRequestTask(Vector2i mapSource, Vector2i mapGoal, PROPULSION_TYPE propulsion)
                    : mapSource(mapSource), mapGoal(mapGoal), propulsion(propulsion) {
            }

            void runPromised() override;

            ~PathRequestTask() override = default;

        private:
            const Vector2i mapSource;
            const Vector2i mapGoal;
            const PROPULSION_TYPE propulsion{};

            unsigned _debugCacheHits = 0;
            unsigned _debugCacheMisses = 0;

            std::vector<std::pair<unsigned, unsigned>>
            portalWalker(unsigned sourcePortalId, unsigned goalPortalId);

            std::vector<std::future<bool>>
            scheduleFlowFields(const std::vector<std::pair<unsigned, unsigned>> &path);
        };

        class FlowFieldSector final {
        public:
            typedef std::array<std::array<Vector2f, SECTOR_SIZE>, SECTOR_SIZE> vectorArrayT;

            FlowFieldSector() = default;

            FlowFieldSector &operator=(FlowFieldSector &) = delete;

            FlowFieldSector &operator=(FlowFieldSector &&) = delete;

            FlowFieldSector(FlowFieldSector &) = delete;

            FlowFieldSector(FlowFieldSector &&) = default;

            ~FlowFieldSector() = default;

            void setVector(unsigned x, unsigned y, Vector2f vector);

            Vector2f getVector(unsigned x, unsigned y) const;

            Vector2f getVector(Vector2f p) const;

        private:
            vectorArrayT vectors;
        };

        // Promise states whether flow field calculation has completed for given sector
        class FlowfieldCalcTask : public FuturedTask<bool> {
        public:
            // Takes goals by copy. Need to control lifetime of goals myself (goals can be constructed ad-hoc)
            FlowfieldCalcTask(Portal::pointsT goals, portalMapT &portals, const sectorListT &sectors,
                              PROPULSION_TYPE propulsion);

            void runPromised() override;

            ~FlowfieldCalcTask() override = default;

        private:
            struct Node {
                unsigned short predecessorCost;
                unsigned index;

                bool operator<(const Node &other) const {
                    // We want top element to have lowest cost
                    return predecessorCost > other.predecessorCost;
                }
            };

            Sector integrationField;
            FlowFieldSector flowField;

            // Constructor depends on member init order
            const Portal::pointsT goals;
            const sectorListT &sectors;
            const unsigned sectorId;
            const AbstractSector &sector;
            PROPULSION_TYPE propulsion;

            void calculateIntegrationField(const Portal::pointsT &points);

            void integratePoints(std::priority_queue<Node> &openSet);

            void calculateFlowField();

            float getCostOrExtrapolate(int x, int y, unsigned short currentTileCost, int neighborX, int neighborY);
        };

        void initCostFields();

        void costFieldReplaceWithEmpty(sectorListT &sectors);

        void setupPortals();

        portalMapT setupPortalsForSectors(sectorListT &sectors);

        void destroyCostFields();

        void destroyPortals();

        void destroyFlowfieldCache();

        Tile createTile(unsigned x, unsigned y, PROPULSION_TYPE propulsion);

        /**
            * Portal detection, by axis.
            * axisStart - starting point for one of the axis
            * otherAxis1 - value from other axis, belonging to thisSector
            * otherAxis2 - value from other axis, belonging to otherSector
            * axisEndOut - last field checked for portal existence. Used as starting point in next iteration
            * If axisStart is "x", then "otherAxis" is y.
            */
        Portal
        detectPortalByAxis(unsigned axisStart, unsigned axisEnd, unsigned otherAxis1, unsigned otherAxis2,
                           bool isXAxis, AbstractSector &thisSector, AbstractSector &otherSector,
                           unsigned &axisEndOut);

        void connectPortals(portalMapT &portalMap, sectorListT &sectors);

        void connectPotentialNeighbor(std::pair<const unsigned, Portal> &portalWithIndex, unsigned potentialNeighbor,
                                      TileAStar &pathfinder, portalMapT &portalMap, bool isSectorEmpty);

        unsigned pointToIndex(Vector2i p);

        Vector2i getPointByFlatIndex(unsigned index);

        unsigned straightLineDistance(Vector2i source, Vector2i destination);

        std::pair<unsigned, unsigned> mapSourceGoalToPortals(Vector2i mapSource, Vector2i mapGoal, PROPULSION_TYPE propulsion);

        /*
            * Helps to decide if we should use firstSectorPoints or secondSectorPoints as goal points
            * See Portal class for more explanation
            */
        bool isForward(Vector2i source, Vector2i firstSectorGoal, Vector2i secondSectorGoal);

        std::vector<std::pair<unsigned, unsigned>> getPathFromCache(unsigned sourcePortalId, unsigned goalPortalId, PROPULSION_TYPE propulsion);

        Portal::pointsT portalToGoals(const Portal &portal, Vector2i currentPosition);

        Vector2f getMovementVector(unsigned nextPortalId, Vector2i currentPosition, PROPULSION_TYPE propulsion);

        Vector2f getMovementVectorToFinalGoals(const Portal::pointsT& goals, Vector2i currentPosition, PROPULSION_TYPE propulsion);

        Vector2f getMovementVector(const Portal::pointsT& goals, Vector2i currentPosition, PROPULSION_TYPE propulsion);

        unsigned getCurrentSectorId(Vector2i currentPosition);

        int sectorIdByTwoPortalIds(unsigned firstPortalId, unsigned secondPortalId, PROPULSION_TYPE propulsion);

        std::vector<Vector2i> portalPathToCoordsPath(const std::vector<std::pair<unsigned, unsigned>> &path,
                                                     PROPULSION_TYPE propulsion);

        //////////////////////////////////////////////////////////////////////////////////////////
        // +- x axis tile debug draw. Smaller values = less tiles drawn. "7" somewhat fits the default window resolution
        constexpr const unsigned DEBUG_DRAW_X_DELTA = 7;
        // +- y axis tile debug draw
        constexpr const unsigned DEBUG_DRAW_Y_DELTA = 6;

        void debugDrawCoords();

        void debugDrawCostField();

        void debugTileDrawCost(AbstractSector &sector, unsigned x, unsigned y, unsigned screenX, unsigned screenY);

        void debugDrawPortals();

        void debugDrawPortalPath();

        void debugDrawIntegrationField();

        void debugDrawFlowField();
    }

}

// Functions needed for sector-level flowfield QCache. Must be declared before QCache
uint qHash(const Vector2i& key, uint seed = 1) {
    return seed * ::qHash(key.x) ^ ::qHash(key.y);
}

uint qHash(const flowfield::detail::Portal::pointsT& key, uint seed = 1) {
    uint hash = seed;

    for (auto& point : key) {
        hash ^= qHash(point);
    }

    return hash;
}

#include <qcache.h>

namespace flowfield {
	static bool flowfieldEnabled = false;

	void enable() {
		flowfieldEnabled = true;
	}

	bool isEnabled() {
		return flowfieldEnabled;
	}

	void init() {
		if (!isEnabled()) return;

		detail::initCostFields();
		detail::setupPortals();

		QThreadPool::globalInstance()->setExpiryTimeout(detail::THREAD_POOL_EXPIRY_TIMEOUT_MS);
		detail::portalPathThreadPool.setExpiryTimeout(detail::THREAD_POOL_EXPIRY_TIMEOUT_MS);

		/*
		* By default, QThreadPool sets maxThreadCount to number of logical cores, or 1 if cannot detect it.
		* One thread is busy with all the game logic. That would allow to effectively use one less thread.
		*/
		if (DEBUG_THREAD_POOL) {
			QThreadPool::globalInstance()->setMaxThreadCount(1);
			detail::portalPathThreadPool.setMaxThreadCount(1);
		} else {
			int maxThreads = std::max(1, QThread::idealThreadCount() - 1);
			QThreadPool::globalInstance()->setMaxThreadCount(maxThreads);

			detail::portalPathThreadPool.setMaxThreadCount(detail::PORTAL_PATH_THREAD_POOL_MAX);
		}
	}

	void destroy() {
		if (!isEnabled()) return;

		detail::destroyCostFields();
		detail::destroyPortals();
		detail::destroyFlowfieldCache();
	}

	void calculateFlowFieldsAsync(MOVE_CONTROL * psMove, unsigned id, int startX, int startY, int tX, int tY, PROPULSION_TYPE propulsionType,
								  DROID_TYPE droidType, FPATH_MOVETYPE moveType, int owner, bool acceptNearest, StructureBounds const & dstStructure) {
		Vector2i source { map_coord(startX), map_coord(startY) };
		Vector2i goal { map_coord(tX), map_coord(tY) };

		auto task = std::make_unique<detail::PathRequestTask>(source, goal, propulsionType);
		std::future<bool> pathRequestFuture = task->getFuture();
		detail::portalPathThreadPool.start(task.release());
	}

	std::vector<std::pair<unsigned, unsigned>> getPathFromCache(unsigned startX, unsigned startY, unsigned tX, unsigned tY, PROPULSION_TYPE propulsion) {
		Vector2i source { map_coord(startX), map_coord(startY) };
		Vector2i goal { map_coord(tX), map_coord(tY) };

		unsigned sourcePortalId, goalPortalId;
		std::tie(sourcePortalId, goalPortalId) = detail::mapSourceGoalToPortals(source, goal, propulsion);

		return detail::getPathFromCache(sourcePortalId, goalPortalId, propulsion);
	}

	Vector2f getMovementVector(unsigned nextPortalId, unsigned currentX, unsigned currentY, PROPULSION_TYPE propulsion) {
		Vector2i position { map_coord(currentX), map_coord(currentY) };
		return detail::getMovementVector(nextPortalId, position, propulsion);
	}

	Vector2f getMovementVectorToFinalGoal(unsigned tX, unsigned tY, unsigned currentX, unsigned currentY, PROPULSION_TYPE propulsion) {
		detail::Portal::pointsT goal{ { map_coord(tX), map_coord(tY) } };
		Vector2i position{ map_coord(currentX), map_coord(currentY) };
		return detail::getMovementVectorToFinalGoals(goal, position, propulsion);
	}

	unsigned getCurrentSectorId(unsigned currentX, unsigned currentY) {
		Vector2i position { map_coord(currentX), map_coord(currentY) };
		return detail::getCurrentSectorId(position);
	}

	std::vector<Vector2i> portalPathToCoordsPath(const std::vector<std::pair<unsigned, unsigned>>& path, PROPULSION_TYPE propulsion) {
		return detail::portalPathToCoordsPath(path, propulsion);
	}

	void debugDraw() {
		if (!isEnabled()) return;

		detail::debugDrawCoords();

		if (COST_FIELD_DEBUG) {
			detail::debugDrawCostField();
		}

		if (PORTALS_DEBUG) {
			detail::debugDrawPortals();
		}

		if (PORTAL_PATH_DEBUG) {
			detail::debugDrawPortalPath();
		}

		if (INTEGRATION_FIELD_DEBUG) {
			detail::debugDrawIntegrationField();
		}

		if (VECTOR_FIELD_DEBUG) {
			detail::debugDrawFlowField();
		}
	}

	namespace detail
	{
		// Propulsion mapping FOR READING DATA ONLY! See below.
		const std::map<PROPULSION_TYPE, unsigned> propulsionToIndex
		{
			// All these share the same flowfield, because they are different types of ground-only
			{PROPULSION_TYPE_WHEELED, 0},
			{PROPULSION_TYPE_TRACKED, 0},
			{PROPULSION_TYPE_LEGGED, 0},
			{PROPULSION_TYPE_HALF_TRACKED, 0},
			//////////////////////////////////
			{PROPULSION_TYPE_PROPELLOR, 1},
			{PROPULSION_TYPE_HOVER, 2},
			{PROPULSION_TYPE_LIFT, 3}
		};

		// Propulsion used in for-loops FOR WRITING DATA. We don't want to process "0" index multiple times.
		const std::map<PROPULSION_TYPE, unsigned> propulsionToIndexUnique
		{
			{PROPULSION_TYPE_WHEELED, 0},
			{PROPULSION_TYPE_PROPELLOR, 1},
			{PROPULSION_TYPE_HOVER, 2},
			{PROPULSION_TYPE_LIFT, 3}
		};

		// Cost fields for ground, hover and lift movement types
		std::array<sectorListT, 4> costFields;

		// Portals connecting sectors
		std::array<portalMapT, 4> portalArr;

		// Mutex for portal-level A* path cache
		std::mutex portalPathMutex;

		// Mutex for sector-level vector field cache
		std::mutex flowfieldMutex;

		// Caches
		typedef QCache<Vector2i, std::vector<std::pair<unsigned, unsigned>>> portalPathCacheT;
		typedef QCache<Portal::pointsT, FlowFieldSector> flowfieldCacheT;

		// Workaround because QCache is neither copyable nor movable
		std::array<std::unique_ptr<portalPathCacheT>, 4> portalPathCache {
			std::make_unique<portalPathCacheT>(PORTAL_PATH_CACHE_MAX),
			std::make_unique<portalPathCacheT>(PORTAL_PATH_CACHE_MAX),
			std::make_unique<portalPathCacheT>(PORTAL_PATH_CACHE_MAX),
			std::make_unique<portalPathCacheT>(PORTAL_PATH_CACHE_MAX)
		};

		std::array<std::unique_ptr<flowfieldCacheT>, 4> flowfieldCache {
			std::make_unique<flowfieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<flowfieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<flowfieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<flowfieldCacheT>(FLOWFIELD_CACHE_MAX)
		};

		constexpr const Tile emptyTile{ COST_MIN };

#ifndef NDEBUG
		unsigned _debugTotalSectors = 0;
		unsigned _debugEmptySectors = 0;

		std::vector<std::pair<unsigned, unsigned>> _debugPortalPath;

		typedef QCache<Portal::pointsT, AbstractSector> _debugintegrationFieldCacheT;
		std::array<std::unique_ptr<_debugintegrationFieldCacheT>, 4> _debugIntegrationFieldCache{
			std::make_unique<_debugintegrationFieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<_debugintegrationFieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<_debugintegrationFieldCacheT>(FLOWFIELD_CACHE_MAX),
			std::make_unique<_debugintegrationFieldCacheT>(FLOWFIELD_CACHE_MAX)
		};
#endif

		//////////////////////////////////////////////////////////////////////////////////////

		bool Tile::isBlocking() const
		{
			return cost == NOT_PASSABLE;
		}

		Portal::Portal(AbstractSector* sector1, AbstractSector* sector2, pointsT& firstSectorPoints, pointsT& secondSectorPoints)
			: firstSector(sector1),	secondSector(sector2), firstSectorPoints(firstSectorPoints), secondSectorPoints(secondSectorPoints)
		{
			assert(firstSectorPoints.size() <= SECTOR_SIZE);
			assert(secondSectorPoints.size() <= SECTOR_SIZE);
		}

		bool Portal::isValid() const
		{
			return !firstSectorPoints.empty();
		}

		Vector2i Portal::getFirstSectorCenter() const {
			return firstSectorPoints[firstSectorPoints.size() / 2];
		}

		Vector2i Portal::getSecondSectorCenter() const {
			return secondSectorPoints[secondSectorPoints.size() / 2];
		}

		unsigned AbstractSector::getIdByCoords(unsigned x, unsigned y, unsigned mapWidth)
		{
			const unsigned xNumber = x / SECTOR_SIZE;
			const unsigned yNumber = y / SECTOR_SIZE;
			const auto sectorsPerRow = mapWidth / SECTOR_SIZE;
			const unsigned sectorId = yNumber * sectorsPerRow + xNumber;

			assert(sectorId < (mapWidth * mapHeight / (SECTOR_SIZE * SECTOR_SIZE)) && "Sector id too big");

			return sectorId;
		}

		unsigned AbstractSector::getIdByCoords(unsigned x, unsigned y)
		{
			return getIdByCoords(x, y, mapWidth);
		}

		unsigned AbstractSector::getIdByCoords(Vector2i p) {
			return getIdByCoords(p.x, p.y);
		}

		Vector2i AbstractSector::getTopLeftCorner(unsigned id)
		{
			assert(id < (mapWidth * mapHeight / (SECTOR_SIZE * SECTOR_SIZE)) && "Sector id too big");

			const auto sectorsPerRow = mapWidth / SECTOR_SIZE;
			const unsigned y = (id / sectorsPerRow) * SECTOR_SIZE;
			const unsigned x = (id % sectorsPerRow) * SECTOR_SIZE;
			return Vector2i{x, y};
		}

		Vector2i AbstractSector::getTopLeftCornerByCoords(Vector2i point) {
			const unsigned sectorId = AbstractSector::getIdByCoords(point);
			return AbstractSector::getTopLeftCorner(sectorId);
		}

		std::vector<unsigned> AbstractSector::getNeighbors(const std::vector<std::unique_ptr<AbstractSector>>& sectors, Vector2i center) {
			assert(center.x < mapWidth);
			assert(center.y < mapHeight);
			std::vector<unsigned> neighbors;
			const unsigned sectorId = AbstractSector::getIdByCoords(center);

			for (int y = -1; y <= 1; y++) {
				const int realY = center.y + y;
				for (int x = -1; x <= 1; x++) {
					const int realX = center.x + x;
					if ((y == 0 && x == 0) || realY < 0 || realX < 0 || realY >= mapHeight || realX >= mapWidth) {
						// Skip self and out-of-map
						continue;
					}

					const unsigned targetSectorId = AbstractSector::getIdByCoords(realX, realY);
					const bool isPassable = !sectors[targetSectorId]->getTile(realX, realY).isBlocking();
					if (sectorId == targetSectorId && isPassable) {
						neighbors.push_back(realY * mapWidth + realX);
					}
				}
			}

			return neighbors;
		}

		bool AbstractSector::isEmpty() const
		{
			return false;
		}

		void AbstractSector::addPortal(unsigned portalId)
		{
			ASSERT(std::find(portalIds.begin(), portalIds.end(), portalId) == portalIds.end(), "Portal in sector already exists");
			portalIds.push_back(portalId);
		}

		const std::vector<unsigned>& AbstractSector::getPortals() const
		{
			return portalIds;
		}

		void Sector::setTile(unsigned x, unsigned y, Tile tile)
		{
			this->tiles[x % SECTOR_SIZE][y % SECTOR_SIZE] = tile;
		}

		Tile Sector::getTile(unsigned x, unsigned y) const
		{
			return this->tiles[x % SECTOR_SIZE][y % SECTOR_SIZE];
		}

		Tile Sector::getTile(Vector2i p) const
		{
			assert(p.x < mapWidth);
			assert(p.y < mapHeight);
			return getTile(p.x, p.y);
		}

		bool Sector::checkIsEmpty() const
		{
			for (auto&& row : this->tiles)
			{
				for (auto&& tile : row)
				{
					if (tile.cost != COST_MIN) return false;
				}
			}

			return true;
		}

		void EmptySector::setTile(unsigned x, unsigned y, Tile tile)
		{
			// No-op - sector is already empty
		}

		Tile EmptySector::getTile(unsigned x, unsigned y) const
		{
			return emptyTile;
		}

		Tile EmptySector::getTile(Vector2i p) const
		{
			return emptyTile;
		}

		bool EmptySector::checkIsEmpty() const
		{
			return true;
		}

		bool EmptySector::isEmpty() const
		{
			return true;
		}

		std::deque<unsigned> AbstractAStar::findPath(unsigned startingIndex, unsigned nodes)
		{
			if (findPathExists(startingIndex, nodes))
			{
				auto path = reconstructPath(startingIndex);
				logDebugNodesStats(nodes, static_cast<unsigned>(path.size()));
				return path;
			}

			// Empty container, no path found for given goal
			return {};
		}

		bool AbstractAStar::findPathExists(unsigned startingIndex, unsigned nodes)
		{
			// AKA closed set
			std::vector<bool> visited(nodes, false);

			// AKA open set
			std::vector<Node> considered{Node{startingIndex, COST_MIN, heuristic(startingIndex)}};

			while (!considered.empty())
			{
				std::pop_heap(considered.begin(), considered.end());
				const auto current = considered.back();
				considered.pop_back();
				visited[current.index] = true;

				if (DEBUG_A_STAR) {
					_debugNodesVisited++;
				}

				if (current.index == goal)
				{
					return true;
				}

				for (unsigned neighbor : getNeighbors(current.index))
				{
					if (visited[neighbor])
					{
						// Ignore the neighbor which is already evaluated.
						continue;
					}

					const unsigned cost = current.cost + distance(current.index, neighbor);

					const auto neighbourIt = std::find_if(considered.begin(), considered.end(), [=](const Node& n)
					{
						return n.index == neighbor;
					});
					if (neighbourIt == considered.end() || cost < neighbourIt->cost)
					{
						cameFrom[neighbor] = current.index;
						if (neighbourIt != considered.end())
						{
							// Updating with new value requires deleting and re-inserting
							considered.erase(neighbourIt);
						}
						considered.push_back(Node{neighbor, cost, heuristic(neighbor)});
						std::push_heap(considered.begin(), considered.end());
					}
				}
			}

			return false;
		}

		std::deque<unsigned> AbstractAStar::reconstructPath(unsigned start)
		{
			std::deque<unsigned> path;
			unsigned current = goal;

			while (current != start)
			{
				path.push_front(current);
				current = cameFrom[current];
			}

			path.push_front(start);
			return path;
		}

		void AbstractAStar::logDebugNodesStats(unsigned nodesTotal, unsigned nodesInPath) const {
			if (DEBUG_A_STAR) {
				std::lock_guard<std::mutex> lock(logMutex);
                wz_debug(LOG_FLOWFIELD, "Nodes total: %d, nodes in path: %d, nodes visited: %d\n", nodesTotal, nodesInPath, _debugNodesVisited);
			}
		}

		TileAStar::TileAStar(unsigned goal, const sectorListT& sectors) : AbstractAStar(goal), sectors(sectors)
		{
			assert(goal < mapWidth * mapHeight);
			goalPoint = getPointByFlatIndex(goal);
			sectorId = AbstractSector::getIdByCoords(goalPoint);
		}

		bool TileAStar::findPathExists(unsigned startingIndex, unsigned nodes)
		{
			assert(startingIndex < mapWidth * mapHeight);

			unsigned startSectorId = AbstractSector::getIdByCoords(getPointByFlatIndex(startingIndex));

			if (startSectorId != sectorId) {
				return false;
			}

			return AbstractAStar::findPathExists(startingIndex, nodes);
		}

		std::vector<unsigned> TileAStar::getNeighbors(unsigned index)
		{
			assert(index < mapWidth * mapHeight);
			const Vector2i currentPoint = getPointByFlatIndex(index);

			return AbstractSector::getNeighbors(sectors, currentPoint);
		}

		unsigned TileAStar::distance(unsigned current, unsigned neighbor)
		{
			assert(current < mapWidth * mapHeight);
			assert(neighbor < mapWidth * mapHeight);
			const Vector2i currentPoint = getPointByFlatIndex(current);
			const Vector2i neighborPoint = getPointByFlatIndex(neighbor);

			return distanceCommon(currentPoint, neighborPoint, COST_MIN);
		}

		unsigned TileAStar::heuristic(unsigned start)
		{
			assert(start < mapWidth * mapHeight);
			const Vector2i startPoint = getPointByFlatIndex(start);
			const unsigned cost = sectors[sectorId]->getTile(startPoint).cost;
			return distanceCommon(startPoint, goalPoint, cost);
		}

		unsigned TileAStar::distanceCommon(Vector2i point1, Vector2i point2, unsigned cost)
		{
			const unsigned dx = std::abs(static_cast<int>(point1.x) - static_cast<int>(point2.x));
			const unsigned dy = abs(static_cast<int>(point1.y) - static_cast<int>(point2.y));

			// Avoid sqrt(2) for diagonal by scaling cost
			const unsigned simpleCost = 10 * cost;
			const unsigned diagonalCost = 14 * cost;

			return simpleCost * (dx + dy) + (diagonalCost - 2 * simpleCost) * std::min(dx, dy);
		}

		std::vector<unsigned> PortalAStar::getNeighbors(unsigned index)
		{
			assert(portals.find(index) != portals.end() && "Portal does not exists");
			Portal& portal = portals[index];
			return portal.neighbors;
		}

		unsigned PortalAStar::distance(unsigned current, unsigned neighbor)
		{
			assert(portals.find(current) != portals.end() && "Portal does not exists");
			assert(portals.find(neighbor) != portals.end() && "Portal does not exists");
			Portal& currentPortal = portals[current];
			Portal& neighborPortal = portals[neighbor];
			return distanceCommon(currentPortal, neighborPortal);
		}

		unsigned PortalAStar::heuristic(unsigned start)
		{
			assert(portals.find(start) != portals.end() && "Portal does not exists");
			Portal& currentPortal = portals[start];
			return distanceCommon(currentPortal, goalPortal);
		}

		unsigned PortalAStar::distanceCommon(const Portal& portal1, const Portal& portal2)
		{
			return straightLineDistance(portal1.getFirstSectorCenter(), portal2.getFirstSectorCenter());
		}

		void PathRequestTask::runPromised() {
			if (DEBUG_BUILD) // Mutex is expensive and won't be optimized in release mode
			{
				std::lock_guard<std::mutex> lock(logMutex);
                wz_debug(LOG_FLOWFIELD, "Path request calculation for start x: %d, y: %d, goal x: %d, y: %d, propulsion: %d\n",
					  mapSource.x, mapSource.y, mapGoal.x, mapGoal.y, propulsion);
			}

			int sourcePortalId, goalPortalId;
			std::tie(sourcePortalId, goalPortalId) = mapSourceGoalToPortals(mapSource, mapGoal, propulsion);

			const auto path = portalWalker(sourcePortalId, goalPortalId);

			if (DEBUG_BUILD) // Mutex is expensive and won't be optimized in release mode
			{
				std::lock_guard<std::mutex> lock(logMutex);
				_debugPortalPath = path;
			}

			auto flowFieldFutures = scheduleFlowFields(path);

			// Wait for all flow field task to complete
			for (auto&& future : flowFieldFutures) {
				future.wait();
			}

			setPromise(true);
		}

		std::vector<std::pair<unsigned, unsigned>> PathRequestTask::portalWalker(unsigned sourcePortalId, unsigned goalPortalId) {

			std::unique_lock<std::mutex> lock(portalPathMutex);
			auto& localPortalPathCache = *portalPathCache[propulsionToIndex.at(propulsion)];
			auto* pathPtr = localPortalPathCache[{sourcePortalId, goalPortalId}];
			if (pathPtr) {
				if (DEBUG_BUILD) // Mutex is expensive and won't be optimized in release mode
				{
					std::lock_guard<std::mutex> logLock(logMutex);
                    wz_debug(LOG_FLOWFIELD, "Flowfield portal cache hit\n");
				}

				return *pathPtr;
			} else {
				lock.unlock();

				if (DEBUG_BUILD) // Mutex is expensive and won't be optimized in release mode
				{
					std::lock_guard<std::mutex> logLock(portalPathMutex);
					wz_debug(LOG_FLOWFIELD, "Flowfield portal cache miss\n");
				}

				portalMapT& portals = portalArr[propulsionToIndex.at(propulsion)];
				PortalAStar portalWalker(goalPortalId, portals);
				const std::deque<unsigned> path = portalWalker.findPath(sourcePortalId, static_cast<unsigned>(portals.size()));

				auto sectorToPortalPath = std::make_unique<std::vector<std::pair<unsigned, unsigned>>>();
				unsigned sectorId = getCurrentSectorId(mapSource);

				// First portal in the path
				sectorToPortalPath->emplace_back(sectorId, sourcePortalId);

				auto pathLength = path.size();
				for (size_t i = 1; i < pathLength; i++) {
					auto prevPortalId = path[i - 1];
					auto portalId = path[i];
					auto sectorIdByPortals = sectorIdByTwoPortalIds(prevPortalId, portalId, propulsion);
					ASSERT(sectorIdByPortals != -1, "Two non-neighboring portal ids: %d, %d", prevPortalId, portalId);
					sectorToPortalPath->emplace_back(sectorIdByPortals, portalId);
				}

				auto sectorToPortalMapCopy(*sectorToPortalPath);
				lock.lock();
				localPortalPathCache.insert({ sourcePortalId, goalPortalId }, sectorToPortalPath.release());
				return sectorToPortalMapCopy;
			}
		}

		std::vector<std::future<bool>> PathRequestTask::scheduleFlowFields(const std::vector<std::pair<unsigned, unsigned>> & path) {
			auto& portals = portalArr[propulsionToIndex.at(propulsion)];
			auto& sectors = costFields[propulsionToIndex.at(propulsion)];
			auto& localFlowFieldCache = *flowfieldCache[propulsionToIndex.at(propulsion)];
			std::vector<std::future<bool>> flowFieldFutures;

			Vector2i localStartPoint = mapSource;

			// Lock the whole loop, but hopefully this will help with lock contention that would otherwise occur
			std::lock_guard<std::mutex> lock(flowfieldMutex);

			for (auto& pair : path) {
				Portal& leavePortal = portals[pair.second];
				Portal::pointsT goals = portalToGoals(leavePortal, localStartPoint);
				localStartPoint = goals[0];

				if (!localFlowFieldCache.contains(goals)) {
					_debugCacheMisses++;

					auto task = std::make_unique<FlowfieldCalcTask>(goals, portals, sectors, propulsion);
					flowFieldFutures.push_back(task->getFuture());
					QThreadPool::globalInstance()->start(task.release());
				} else {
					_debugCacheHits++;
				}
			}

			// Final goal task
			// TODO: in future with better integration with Warzone, there might be multiple goals for a formation, so droids don't bump into each other
			Portal::pointsT finalGoals { mapGoal };

			if (!localFlowFieldCache.contains(finalGoals)) {
				_debugCacheMisses++;

				auto task = std::make_unique<FlowfieldCalcTask>(finalGoals, portals, sectors, propulsion);
				flowFieldFutures.push_back(task->getFuture());
				QThreadPool::globalInstance()->start(task.release());
			} else {
				_debugCacheHits++;
			}

			if (DEBUG_BUILD) // Mutex is expensive and won't be optimized in release mode
			{
				std::lock_guard<std::mutex> logLock(logMutex);
				wz_debug(LOG_FLOWFIELD, "Flowfield sector cache hits: %d, misses: %d\n", _debugCacheHits, _debugCacheMisses);
			}

			return flowFieldFutures;
		}

		void FlowFieldSector::setVector(unsigned x, unsigned y, Vector2f vector) {
			vectors[x][y] = vector;
		}

        Vector2f FlowFieldSector::getVector(unsigned x, unsigned y) const {
			return vectors[x % SECTOR_SIZE][y % SECTOR_SIZE];
		}

        Vector2f FlowFieldSector::getVector(Vector2f p) const {
			return getVector(p.x, p.y);
		}

		FlowfieldCalcTask::FlowfieldCalcTask(Portal::pointsT goals, portalMapT& portals, const sectorListT& sectors, PROPULSION_TYPE propulsion)
			: goals(goals), sectors(sectors), sectorId(AbstractSector::getIdByCoords(*goals.begin())),
			sector(*sectors[sectorId]), propulsion(propulsion) {
		}

		void FlowfieldCalcTask::runPromised() {
			// TODO: maybe convert to task calculating all the path, so values from previous sectors are added at sector borders
			// TODO: maybe smoothing it with bilinear will be enough (when reading)? https://howtorts.github.io/2014/01/04/basic-flow-fields.html
			calculateIntegrationField(goals);

			calculateFlowField();

			{
				std::lock_guard<std::mutex> lock(flowfieldMutex);
				auto flowfieldMoved = std::make_unique<FlowFieldSector>(std::move(flowField));
				flowfieldCache[propulsionToIndex.at(propulsion)]->insert(goals, flowfieldMoved.release());

				if (INTEGRATION_FIELD_DEBUG) {
					auto integrationFieldMoved = std::make_unique<Sector>(std::move(integrationField));
					_debugIntegrationFieldCache[propulsionToIndex.at(propulsion)]->insert(goals, integrationFieldMoved.release());
				}
			}

			setPromise(true);
		}

		void FlowfieldCalcTask::calculateIntegrationField(const Portal::pointsT& points) {
			// TODO: here do checking if given tile contains a building (instead of doing that in cost field)
			// TODO: split NOT_PASSABLE into a few constants, for terrain, buildings and maybe sth else
			for (unsigned x = 0; x < SECTOR_SIZE; x++) {
				for (unsigned y = 0; y < SECTOR_SIZE; y++) {
					integrationField.setTile(x, y, Tile { NOT_PASSABLE });
				}
			}

			// Thanks to priority queue, we have "water pouring effect".
			// First we go where cost is the lowest, so we don't discover better path later.
			std::priority_queue<Node> openSet;

			for (auto& point : points) {
				openSet.push({ 0, pointToIndex(point) });
			}

			while (!openSet.empty()) {
				integratePoints(openSet);
				openSet.pop();
			}
		}

		void FlowfieldCalcTask::integratePoints(std::priority_queue<Node>& openSet) {
			const Node& node = openSet.top();
			Vector2i nodePoint = getPointByFlatIndex(node.index);
			Tile nodeTile = sector.getTile(nodePoint);

			if (nodeTile.isBlocking()) {
				return;
			}

			unsigned short nodeCostFromCostField = nodeTile.cost;

			// Go to the goal, no matter what
			if (node.predecessorCost == 0) {
				nodeCostFromCostField = COST_MIN;
			}

			const unsigned short newCost = node.predecessorCost + nodeCostFromCostField;
			const unsigned short nodeOldCost = integrationField.getTile(nodePoint).cost;

			if (newCost < nodeOldCost) {
				integrationField.setTile(nodePoint.x, nodePoint.y, Tile { newCost });

				for (unsigned neighbor : AbstractSector::getNeighbors(sectors, nodePoint)) {
					openSet.push({ newCost, neighbor });
				}
			}
		}

		void FlowfieldCalcTask::calculateFlowField() {
			for (int y = 0; y < SECTOR_SIZE; y++) {
				for (int x = 0; x < SECTOR_SIZE; x++) {
					Tile tile = integrationField.getTile(x, y);
					if (tile.isBlocking()) {
						// Skip non-passable
						flowField.setVector(x, y, Vector2f { 0.0f, 0.0f });
						continue;
					}

					const float leftCost = getCostOrExtrapolate(x - 1, y, tile.cost, x + 1, y);
					const float rightCost = getCostOrExtrapolate(x + 1, y, tile.cost, x - 1, y);

					const float topCost = getCostOrExtrapolate(x, y - 1, tile.cost, x, y + 1);
					const float bottomCost = getCostOrExtrapolate(x, y + 1, tile.cost, x, y - 1);

                    Vector2f vector{};
					vector.x = leftCost - rightCost;
					vector.y = topCost - bottomCost;
					vector = glm::normalize(vector);

					if (std::abs(vector.x) < 0.01f && std::abs(vector.y) < 0.01f) {
						// Local optima. Tilt the vector in any direction.
						vector.x = 0.1f;
						vector.y = 0.1f;
					}

					flowField.setVector(x, y, vector);
				}
			}
		}

		float FlowfieldCalcTask::getCostOrExtrapolate(int x, int y, unsigned short currentTileCost, int neighborX, int neighborY) {

			bool onSectorBorder = x < 0 || y < 0 || x >= SECTOR_SIZE || y >= SECTOR_SIZE;

			if (onSectorBorder || integrationField.getTile(x, y).isBlocking()) {
				// Follow trend when no cost available (difference between current and neighbor).
				// For example: 4 7 | x, where x is in another sector. Then x = 7 + (7 - 4) = 10, and we get: 4 7 | 10
				auto elseTile = integrationField.getTile(neighborX, neighborY);
				return static_cast<float>(currentTileCost + (currentTileCost - elseTile.cost));
			}

			const Tile tile = integrationField.getTile(x, y);
			return static_cast<float>(tile.cost);
		}

		void initCostFields()
		{
			// Assume map is already loaded. Access globals
			assert(mapWidth % SECTOR_SIZE == 0);
			assert(mapHeight % SECTOR_SIZE == 0);

			const auto numSectors = (mapWidth / SECTOR_SIZE) * (mapHeight / SECTOR_SIZE);

			// Reserve and fill cost fields with empty sectors
			for (auto& sectors : costFields)
			{
				sectors.reserve(numSectors);
				for (int i = 0; i < numSectors; i++)
				{
					sectors.push_back(std::make_unique<Sector>());
				}
			}

			// Fill tiles in sectors
			for (int x = 0; x < mapWidth; x++)
			{
				for (int y = 0; y < mapHeight; y++)
				{
					const unsigned sectorId = Sector::getIdByCoords(x, y);

					for (auto&& propType : propulsionToIndexUnique)
					{
						const Tile groundTile = createTile(x, y, propType.first);
						costFields[propType.second][sectorId]->setTile(x, y, groundTile);
					}
				}
			}

			// Optimization: replace sectors with no cost with empty sector
			for (auto& sectors : costFields)
			{
				costFieldReplaceWithEmpty(sectors);
				_debugTotalSectors += static_cast<unsigned>(sectors.size());
			}
		}

		void costFieldReplaceWithEmpty(sectorListT& sectors)
		{
			for (auto& sector : sectors)
			{
				if (sector->checkIsEmpty())
				{
					sector = std::make_unique<EmptySector>();
					_debugEmptySectors++;
				}
			}
		}

		void setupPortals()
		{
			for (auto&& propType : propulsionToIndexUnique)
			{
				auto portalMap = setupPortalsForSectors(costFields[propType.second]);
				connectPortals(portalMap, costFields[propType.second]);
				portalArr[propType.second] = std::move(portalMap);
			}
		}

		portalMapT setupPortalsForSectors(sectorListT& sectors)
		{
			portalMapT portals;
			const auto sectorsPerRow = mapWidth / SECTOR_SIZE;
			const auto lastRow = sectors.size() - sectorsPerRow;

			const auto portalAppender = [&](Portal& portalByAxis, AbstractSector& thisSector, AbstractSector& otherSector)
			{
				auto index = static_cast<unsigned>(portals.size());
				portals[index] = std::move(portalByAxis);
				thisSector.addPortal(index);
				otherSector.addPortal(index);
			};

			for (unsigned i = 0; i < sectors.size(); i++)
			{
				const auto corner = AbstractSector::getTopLeftCorner(i);
				AbstractSector& thisSector = *sectors[i];
				bool lastPortalWasValid;

				// Bottom. Skip last row
				if (i < lastRow)
				{
					unsigned x = corner.x;
					do
					{
						AbstractSector& otherSector = *sectors[i + sectorsPerRow];
						Portal portalByAxis = detectPortalByAxis(x, corner.x + SECTOR_SIZE, corner.y + SECTOR_SIZE - 1, corner.y + SECTOR_SIZE, true,
																 thisSector, otherSector, x);
						x++;

						lastPortalWasValid = portalByAxis.isValid();
						if (lastPortalWasValid) {
							portalAppender(portalByAxis, thisSector, otherSector);
						}
					} while (lastPortalWasValid);
				}

				// Right. Skip last column
				if (i % sectorsPerRow != sectorsPerRow - 1)
				{
					unsigned y = corner.y;
					do
					{
						AbstractSector& otherSector = *sectors[i + 1];
						Portal portalByAxis = detectPortalByAxis(y, corner.y + SECTOR_SIZE, corner.x + SECTOR_SIZE - 1, corner.x + SECTOR_SIZE, false,
																 thisSector, otherSector, y);
						y++;

						lastPortalWasValid = portalByAxis.isValid();
						if (lastPortalWasValid) {
							portalAppender(portalByAxis, thisSector, otherSector);
						}
					} while (lastPortalWasValid);
				}
			}

			return portals;
		}

		void destroyCostFields()
		{
			for (auto& sectors : costFields)
			{
				sectors.clear();
			}
		}

		void destroyPortals()
		{
			for (auto& portal : portalArr)
			{
				portal.clear();
			}
		}

		void destroyFlowfieldCache() {
			for (auto&& pair : propulsionToIndexUnique) {
				portalPathCache[pair.second]->clear();
				flowfieldCache[pair.second]->clear();
			}
		}

		Tile createTile(unsigned x, unsigned y, PROPULSION_TYPE propulsion)
		{
			assert(x < mapWidth);
			assert(y < mapHeight);
			unsigned short cost = NOT_PASSABLE;
			const bool isBlocking = fpathBlockingTile(x, y, propulsion);

			// TODO: Current impl forbids VTOL from flying over short buildings
			if (!isBlocking)
			{
				int pMax, pMin;
				getTileMaxMin(static_cast<int>(x), static_cast<int>(y), &pMax, &pMin);

				const auto delta = static_cast<unsigned short>(pMax - pMin);

				if (propulsion != PROPULSION_TYPE_LIFT && delta > SLOPE_THRESOLD)
				{
					// Yes, the cost is integer and we do not care about floating point tail
					cost = std::max(COST_MIN, static_cast<unsigned short>(SLOPE_COST_BASE * static_cast<float>(delta)));
				}
				else
				{
					cost = COST_MIN;
				}
			}

			return Tile{cost};
		}

		Portal detectPortalByAxis(unsigned axisStart, unsigned axisEnd, unsigned otherAxis1, unsigned otherAxis2,
		                          bool isXAxis, AbstractSector& thisSector, AbstractSector& otherSector, unsigned& axisEndOut)
		{
			axisEndOut = 0;
			Portal::pointsT firstSectorPoints;
			Portal::pointsT secondSectorPoints;
			Vector2i firstSectorPoint;
			Vector2i secondSectorPoint;

			for (unsigned axis = axisStart; axis < axisEnd; axis++)
			{
				if (isXAxis)
				{
					firstSectorPoint = Vector2i{ axis, otherAxis1 };
					secondSectorPoint = Vector2i{ axis, otherAxis2 };
				}
				else
				{
					firstSectorPoint = Vector2i{ otherAxis1, axis };
					secondSectorPoint = Vector2i{ otherAxis2, axis };
				}

				bool thisPassable = !thisSector.getTile(firstSectorPoint).isBlocking();
				bool otherPassable = !otherSector.getTile(secondSectorPoint).isBlocking();

				if (thisPassable && otherPassable)
				{
					firstSectorPoints.push_back(firstSectorPoint);
					secondSectorPoints.push_back(secondSectorPoint);
					axisEndOut = axis;
				}
				else if (!firstSectorPoints.empty())
				{
					// Not passable, but we found some points - that means we reached end of portal
					break;
				}
			}

			if (!firstSectorPoints.empty())
			{
				return Portal(&thisSector, &otherSector, firstSectorPoints, secondSectorPoints);
			}
			else
			{
				// Invalid portal
				return Portal();
			}
		}

		void connectPortals(portalMapT& portalMap, sectorListT& sectors)
		{
			for (auto& portalWithIndex : portalMap)
			{
				const Portal& portal = portalWithIndex.second;
				TileAStar firstSectorPathfinder(pointToIndex(portal.getFirstSectorCenter()), sectors);
				TileAStar secondSectorPathfinder(pointToIndex(portal.getSecondSectorCenter()), sectors);

				for (unsigned potentialNeighbor : portal.firstSector->getPortals())
				{
					assert(portalMap.find(potentialNeighbor) != portalMap.end() && "Portal does not exists");
					connectPotentialNeighbor(portalWithIndex, potentialNeighbor, firstSectorPathfinder, portalMap, portal.firstSector->isEmpty());
				}
				for (unsigned potentialNeighbor : portal.secondSector->getPortals())
				{
					assert(portalMap.find(potentialNeighbor) != portalMap.end() && "Portal does not exists");
					connectPotentialNeighbor(portalWithIndex, potentialNeighbor, secondSectorPathfinder, portalMap, portal.secondSector->isEmpty());
				}
			}
		}

		unsigned pointToIndex(Vector2i p) {
			return p.y * mapWidth + p.x;
		}

		Vector2i getPointByFlatIndex(unsigned index) {
			assert(index < mapWidth * mapHeight);
			const unsigned y = index / mapWidth;
			const unsigned x = index % mapWidth;
			return Vector2i{ x, y };
		}

		void connectPotentialNeighbor(std::pair<const unsigned, Portal>& portalWithIndex, unsigned potentialNeighbor,
									  TileAStar& pathfinder, portalMapT& portalMap, bool isSectorEmpty)
		{
			bool myself = portalWithIndex.first == potentialNeighbor;
			auto& actualNeighbors = portalWithIndex.second.neighbors;
			bool alreadyProcessed = std::find(actualNeighbors.begin(), actualNeighbors.end(), potentialNeighbor) != actualNeighbors.end();

			if (!myself && !alreadyProcessed)
			{
				if (isSectorEmpty) {
					portalWithIndex.second.neighbors.push_back(potentialNeighbor);
					portalMap[potentialNeighbor].neighbors.push_back(portalWithIndex.first);
					return;
				}

				// We actually don't need "mapWidth * mapHeight" nodes, only SECTOR_SIZE^2, but we use absolute index for tiles
				// It's not much anyways, since it's only used for vector<bool>, which is usually compressed.
				// Worst case: 256^2 =~ 65KB in uncompressed mode, 256^2 / 8 =~ 8KB in compressed mode.
				unsigned nodes = mapWidth * mapHeight;

				Vector2i potentialNeighborPointFirst = portalMap[potentialNeighbor].getFirstSectorCenter();
				Vector2i potentialNeighborPointSecond = portalMap[potentialNeighbor].getSecondSectorCenter();
				bool pathExists = pathfinder.findPathExists(pointToIndex(potentialNeighborPointFirst), nodes)
					|| pathfinder.findPathExists(pointToIndex(potentialNeighborPointSecond), nodes);
				if (pathExists)
				{
					portalWithIndex.second.neighbors.push_back(potentialNeighbor);
					portalMap[potentialNeighbor].neighbors.push_back(portalWithIndex.first);
				}
			}
		}

		unsigned straightLineDistance(Vector2i source, Vector2i destination) {
			const unsigned dx = abs(static_cast<int>(source.x) - static_cast<int>(destination.x));
			const unsigned dy = abs(static_cast<int>(source.y) - static_cast<int>(destination.y));
			return static_cast<unsigned>(iHypot(dx, dy));
		}

		std::pair<unsigned, unsigned> mapSourceGoalToPortals(Vector2i mapSource, Vector2i mapGoal, PROPULSION_TYPE propulsion) {
			const auto sourceSector = AbstractSector::getIdByCoords(mapSource);
			const auto goalSector = AbstractSector::getIdByCoords(mapGoal);
			const auto& sectors = costFields[propulsionToIndex.at(propulsion)];
			const auto& sourcePortals = sectors[sourceSector]->getPortals();
			const auto& goalPortals = sectors[goalSector]->getPortals();

			auto& portals = portalArr[propulsionToIndex.at(propulsion)];

			// Use straight-line distance to select source portal and goal portal
			const auto lessDistance = [&](const Vector2i &source) {
				return [&](const unsigned id1, const unsigned id2) {
					Portal& p1 = portals[id1];
					Portal& p2 = portals[id2];

					const unsigned p1Distance = straightLineDistance(source, p1.getFirstSectorCenter());
					const unsigned p2Distance = straightLineDistance(source, p2.getFirstSectorCenter());
					return p1Distance < p2Distance;
				};
			};

			// Choose source portal which is closest to the goal, so we don't go back in case when "back" portal is closest to us
			const auto sourcePortalId = *std::min_element(sourcePortals.begin(), sourcePortals.end(), lessDistance(mapGoal));
			// The same applies for the goal portal
			const auto goalPortalId = *std::min_element(goalPortals.begin(), goalPortals.end(), lessDistance(mapSource));

			return { sourcePortalId , goalPortalId };
		}

		bool isForward(Vector2i source, Vector2i firstSectorGoal, Vector2i secondSectorGoal) {
			const Vector2i sourceTlCorner = AbstractSector::getTopLeftCornerByCoords(source);
			const Vector2i firstSectorTlCorner = AbstractSector::getTopLeftCornerByCoords(firstSectorGoal);
			const Vector2i secondSectorTlCorner = AbstractSector::getTopLeftCornerByCoords(secondSectorGoal);

			const unsigned distFirst = straightLineDistance(sourceTlCorner, firstSectorTlCorner);
			const unsigned distSecond = straightLineDistance(sourceTlCorner, secondSectorTlCorner);

			return distFirst < distSecond;
		}

		std::vector<std::pair<unsigned, unsigned>> getPathFromCache(unsigned sourcePortalId, unsigned goalPortalId, PROPULSION_TYPE propulsion) {

			std::lock_guard<std::mutex> lock(portalPathMutex);
			auto& localPortalPathCache = *portalPathCache[propulsionToIndex.at(propulsion)];
			auto* pathPtr = localPortalPathCache[{sourcePortalId, goalPortalId}];
			if (pathPtr) {
				return *pathPtr;
			}

			return {};
		}

		Portal::pointsT portalToGoals(const Portal& portal, Vector2i currentPosition) {
			const bool forward = isForward(currentPosition, portal.getFirstSectorCenter(), portal.getSecondSectorCenter());

			if (forward) {
				return portal.firstSectorPoints;
			} else {
				return portal.secondSectorPoints;
			}
		}

		Vector2f getMovementVector(unsigned nextPortalId, Vector2i currentPosition, PROPULSION_TYPE propulsion) {
			const auto& portals = detail::portalArr[detail::propulsionToIndex.at(propulsion)];
			const detail::Portal& nextPortal = portals.at(nextPortalId);
			Portal::pointsT goals = portalToGoals(nextPortal, currentPosition);

			return getMovementVector(goals, currentPosition, propulsion);
		}

		Vector2f getMovementVectorToFinalGoals(const Portal::pointsT& goals, Vector2i currentPosition, PROPULSION_TYPE propulsion) {
			unsigned propulsionIndex = propulsionToIndex.at(propulsion);
			RayCastCallbackData data{ propulsionIndex, false };
			Vector2i middleGoal = goals[goals.size() / 2];

			rayCast(currentPosition, middleGoal, [](Vector2i pos, int32_t dist, void* data) {
				auto *dataCasted = reinterpret_cast<RayCastCallbackData*>(data);
				auto sectorId = getCurrentSectorId(pos);
				bool isBlocking = costFields[dataCasted->propulsionIndex][sectorId]->getTile(pos).isBlocking();
				dataCasted->isPassable |= !isBlocking;
				return dataCasted->isPassable;
			}, &data);

			if (data.isPassable) {
				// Line of sight, go straight to the target
				Vector2f vector = (middleGoal - currentPosition);
				return vector;
			}

			return getMovementVector(goals, currentPosition, propulsion);
		}

		Vector2f getMovementVector(const Portal::pointsT& goals, Vector2i currentPosition, PROPULSION_TYPE propulsion) {
			unsigned propulsionIndex = propulsionToIndex.at(propulsion);
			std::lock_guard<std::mutex> lock(flowfieldMutex);
			flowfieldCacheT &localFlowfieldCache = *flowfieldCache[propulsionIndex];
			FlowFieldSector *sector = localFlowfieldCache[goals];

			if (sector) {
                Vector2f vector = sector->getVector(currentPosition);
				return { vector.x, vector.y };
			} else {
				// (0,0) vector considered invalid
				return { 0.0f, 0.0f };
			}
		}

		unsigned getCurrentSectorId(Vector2i currentPosition) {
			return AbstractSector::getIdByCoords(currentPosition);
		}

		int sectorIdByTwoPortalIds(unsigned firstPortalId, unsigned secondPortalId, PROPULSION_TYPE propulsion) {
			const auto &portals = detail::portalArr[detail::propulsionToIndex.at(propulsion)];
			const detail::Portal &firstPortal = portals.at(firstPortalId);
			const detail::Portal &secondPortal = portals.at(secondPortalId);

			Vector2i portalCenterInSector{ 0, 0 };
			if (firstPortal.firstSector == secondPortal.secondSector || firstPortal.firstSector == secondPortal.firstSector) {
				portalCenterInSector = firstPortal.getFirstSectorCenter();
			} else if (firstPortal.secondSector == secondPortal.firstSector || firstPortal.secondSector == secondPortal.secondSector) {
				portalCenterInSector = firstPortal.getSecondSectorCenter();
			}

			if (portalCenterInSector.x != 0 && portalCenterInSector.y != 0) {
				return static_cast<int>(AbstractSector::getIdByCoords(portalCenterInSector));
			}

			return -1;
		}

		std::vector<Vector2i> portalPathToCoordsPath(const std::vector<std::pair<unsigned, unsigned>>& path, PROPULSION_TYPE propulsion) {
			auto& portals = portalArr[detail::propulsionToIndex.at(propulsion)];

			std::vector<Vector2i> coordsPath;
			std::transform(path.begin(), path.end(), std::back_inserter(coordsPath), [&](const std::pair<unsigned, unsigned>& pair) {
				// TODO: does it matter, which point it is? Droid will move in the right direction. And portal-level path is only used when no flowfield is available
				auto point = portals[pair.second].getFirstSectorCenter();
				return world_coord(Vector2i(point.x, point.y));
			});

			return coordsPath;
		}

		void debugDrawCoords()
		{
			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const int xDelta = DEBUG_DRAW_X_DELTA;
			const int yDelta = DEBUG_DRAW_Y_DELTA;

			for (int y = -yDelta; y <= yDelta; y++)
			{
				for (int x = -xDelta; x <= xDelta; x++)
				{
					const int actualX = playerXTile + x;
					const int actualY = playerZTile + y;

					if (tileOnMap(actualX, actualY))
					{
						WzText coordsText(std::to_string(actualX) + ", " + std::to_string(actualY), font_small);
						// HACK draw
						const int renderX = 40 + ((x + xDelta) << 6);
						const int renderY = 35 + ((y + yDelta) << 6);
						coordsText.render(renderX, renderY, WZCOL_TEXT_BRIGHT);
					}
				}
			}
		}

		void debugDrawCostField()
		{
			const auto& groundSectors = costFields[propulsionToIndex.at(PROPULSION_TYPE_WHEELED)];
			if (groundSectors.empty()) return;

			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const int xDelta = DEBUG_DRAW_X_DELTA;
			const int yDelta = DEBUG_DRAW_Y_DELTA;

			for (int y = -yDelta; y <= yDelta; y++)
			{
				for (int x = -xDelta; x <= xDelta; x++)
				{
					const int actualX = playerXTile + x;
					const int actualY = playerZTile + y;

					if (tileOnMap(actualX, actualY))
					{
						const unsigned sectorId = Sector::getIdByCoords(actualX, actualY);
						debugTileDrawCost(*groundSectors[sectorId], actualX, actualY, x + xDelta, y + yDelta);
					}
				}
			}
		}

		void debugTileDrawCost(AbstractSector& sector, unsigned x, unsigned y, unsigned screenX, unsigned screenY)
		{
			WzText costText(std::to_string(sector.getTile(x, y).cost), font_medium);
			// HACK
			// I have completely NO IDEA how to draw stuff correctly. This code is by trial-and-error.
			// It's debug only, but it could be a bit better. Too many magic numers, and works only on initial zoom and rotation.
			const int renderX = 40 + (screenX << 6);
			const int renderY = 20 + (screenY << 6);
			costText.render(renderX, renderY, WZCOL_TEXT_BRIGHT);

			const bool topLeftCorner = (x % SECTOR_SIZE == 0) && (y % SECTOR_SIZE == 0);
			const bool bottomLeftCorner = (x % SECTOR_SIZE == 0) && (y % SECTOR_SIZE == SECTOR_SIZE - 1);
			const bool topRightCorner = (x % SECTOR_SIZE == SECTOR_SIZE - 1) && (y % SECTOR_SIZE == 0);
			const bool bottomRightCorner = (x % SECTOR_SIZE == SECTOR_SIZE - 1) && (y % SECTOR_SIZE == SECTOR_SIZE - 1);

			if (topLeftCorner || bottomLeftCorner || topRightCorner || bottomRightCorner)
			{
				iV_Box(renderX, renderY - 14, renderX + 60, renderY + 46, WZCOL_WHITE);
			}
		}

		void debugDrawPortals()
		{
			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const auto convertX = [=](unsigned x)
			{
				return 38 + ((x + (DEBUG_DRAW_X_DELTA - playerXTile)) << 6);
			};

			const auto convertY = [=](unsigned y)
			{
				return 6 + ((y + (DEBUG_DRAW_Y_DELTA - playerZTile)) << 6);
			};

			auto&& portals = portalArr[propulsionToIndex.at(PROPULSION_TYPE_WHEELED)];

			for (auto&& portal : portals)
			{
				iV_Box(convertX(portal.second.getFirstSectorCenter().x), convertY(portal.second.getFirstSectorCenter().y),
				       convertX(portal.second.getSecondSectorCenter().x + 1), convertY(portal.second.getSecondSectorCenter().y + 1), WZCOL_RED);

				WzText costText(std::to_string(portal.first), font_small);
				costText.render(convertX(portal.second.getFirstSectorCenter().x), convertY(portal.second.getFirstSectorCenter().y + 1), WZCOL_RED);

				// Connection with other portals
				for (unsigned neighbor : portal.second.neighbors)
				{
					Portal& neighborPortal = portals[neighbor];
					iV_Line(convertX(portal.second.getFirstSectorCenter().x), convertY(portal.second.getFirstSectorCenter().y),
							convertX(neighborPortal.getSecondSectorCenter().x), convertY(neighborPortal.getSecondSectorCenter().y),
							WZCOL_YELLOW);
				}
			}
		}

		void debugDrawPortalPath() {
			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const auto convertX = [=](unsigned x) {
				return 40 + ((x + (DEBUG_DRAW_X_DELTA - playerXTile)) << 6);
			};

			const auto convertY = [=](unsigned y) {
				return 10 + ((y + (DEBUG_DRAW_Y_DELTA - playerZTile)) << 6);
			};

			auto&& portals = portalArr[propulsionToIndex.at(PROPULSION_TYPE_WHEELED)];

			// It is only debug. If lock happens not to be available, skip drawing
			std::unique_lock<std::mutex> lock(portalPathMutex, std::try_to_lock);
			if (lock) {
				Portal* previousPortal = nullptr;
				for (auto & it : _debugPortalPath) {
					auto& currentPortal = portals[it.second];

					if (previousPortal != nullptr) {
						iV_Line(convertX(previousPortal->getSecondSectorCenter().x), convertY(previousPortal->getSecondSectorCenter().y),
								convertX(currentPortal.getFirstSectorCenter().x), convertY(currentPortal.getFirstSectorCenter().y),
								WZCOL_GREEN);
					}

					iV_Line(convertX(currentPortal.getFirstSectorCenter().x), convertY(currentPortal.getFirstSectorCenter().y),
							convertX(currentPortal.getSecondSectorCenter().x), convertY(currentPortal.getSecondSectorCenter().y),
							WZCOL_GREEN);

					previousPortal = &currentPortal;
				}
				lock.unlock();
			}
		}

		void debugDrawIntegrationField() {
			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const auto convertX = [=](unsigned x) {
				return 40 + ((x + (DEBUG_DRAW_X_DELTA - playerXTile)) << 6);
			};

			const auto convertY = [=](unsigned y) {
				return 50 + ((y + (DEBUG_DRAW_Y_DELTA - playerZTile)) << 6);
			};

			// It is only debug. If lock happens not to be available, skip drawing
			std::unique_lock<std::mutex> lock(flowfieldMutex, std::try_to_lock);
			if (lock) {
				auto &cache = _debugIntegrationFieldCache[propulsionToIndex.at(PROPULSION_TYPE_WHEELED)];
				auto keys = cache->keys();

				for (auto &&key : keys) {
					int goalX = key[0].x;
					int goalY = key[0].y;
					bool onScreen = (std::abs(playerXTile - goalX) < SECTOR_SIZE * 2) && (std::abs(playerZTile - goalY) < SECTOR_SIZE * 2);

					if (onScreen) {
						Vector2i tlCorner = AbstractSector::getTopLeftCornerByCoords(key[0]);

						// Draw vectors
						auto &sector = *cache->object(key);
						for (unsigned y = 0; y < SECTOR_SIZE; y++) {
							for (unsigned x = 0; x < SECTOR_SIZE; x++) {

								auto cost = sector.getTile(x, y).cost;
								const unsigned absoluteX = tlCorner.x + x;
								const unsigned absoluteY = tlCorner.y + y;

								WzText costText(std::to_string(cost), font_medium);
								costText.render(convertX(absoluteX), convertY(absoluteY), WZCOL_TEAM8);
							}
						}
					}
				}

				lock.unlock();
			}
		}

		// TODO: Somewhere there are going 2 different path request even for single vehicle, with one coord differring by few values. Investigate.
		void debugDrawFlowField() {
			const int playerXTile = map_coord(player.p.x);
			const int playerZTile = map_coord(player.p.z);

			const auto convertX = [=](int x) {
				return 60 + ((x + (DEBUG_DRAW_X_DELTA - playerXTile)) << 6);
			};

			const auto convertY = [=](int y) {
				return 30 + ((y + (DEBUG_DRAW_Y_DELTA - playerZTile)) << 6);
			};

			// It is only debug. If lock happens not to be available, skip drawing
			std::unique_lock<std::mutex> lock(flowfieldMutex, std::try_to_lock);
			if (lock) {
				auto& cache = flowfieldCache[propulsionToIndex.at(PROPULSION_TYPE_WHEELED)];
				auto keys = cache->keys();

				for (auto&& key : keys) {
					int goalX = key[0].x;
					int goalY = key[0].y;
					bool onScreen = (std::abs(playerXTile - goalX) < SECTOR_SIZE * 2) && (std::abs(playerZTile - goalY) < SECTOR_SIZE * 2);

					if (onScreen) {
						Vector2i tlCorner = AbstractSector::getTopLeftCornerByCoords(key[0]);

						// Draw goals
						for (auto&& goal : key) {
							iV_Box(convertX(goal.x) - 5, convertY(goal.y) - 5,
								   convertX(goal.x) + 5, convertY(goal.y) + 5,
								   WZCOL_TEAM7);
						}

						// Draw vectors
						auto& sector = *cache->object(key);
						for (unsigned y = 0; y < SECTOR_SIZE; y++) {
							for (unsigned x = 0; x < SECTOR_SIZE; x++) {
								auto vector = sector.getVector(x, y);
								const int absoluteX = tlCorner.x + x;
								const int absoluteY = tlCorner.y + y;

								// Vector direction
								iV_Line(convertX(absoluteX), convertY(absoluteY),
										convertX(absoluteX) + vector.x * std::pow(2, 4), convertY(absoluteY) + vector.y * std::pow(2, 4),
										WZCOL_TEAM2);

								// Vector start point
								iV_ShadowBox(convertX(absoluteX) - 2, convertY(absoluteY) - 2,
											 convertX(absoluteX) + 2, convertY(absoluteY) + 2,
											 0, WZCOL_TEAM7, WZCOL_TEAM7, WZCOL_TEAM7);
							}
						}
					}
				}

				lock.unlock();
			}
		}
	}
}
