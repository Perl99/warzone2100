#include <array>
#include <cmath>
#include <deque>
#include <future>
#include <memory>
#include <queue>

#include "lib/framework/vector.h"
#include <qthreadpool.h>
#include <qelapsedtimer.h>

// Private header
namespace flowfield
{
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
		const unsigned int PORTAL_PATH_THREAD_POOL_MAX = QThread::idealThreadCount();

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
			AbstractSector& operator=(AbstractSector&) = delete;
			AbstractSector& operator=(AbstractSector&&) = delete;
			AbstractSector(AbstractSector&) = delete;
			AbstractSector(AbstractSector&&) = default;
			virtual ~AbstractSector() = default;

			virtual void setTile(const unsigned int x, const unsigned int y, Tile tile) = 0;
			virtual Tile getTile(const unsigned int x, const unsigned int y) const = 0;
			virtual Tile getTile(const Vector2i& p) const = 0;
			virtual bool checkIsEmpty() const = 0; // Actual iterating through tiles
			virtual bool isEmpty() const; // If EmptySector or just Sector
			void addPortal(const unsigned int portalId);
			const std::vector<unsigned int>& getPortals() const;

			static unsigned int getIdByCoords(unsigned int x, unsigned int y, unsigned int mapWidth);
			static unsigned int getIdByCoords(unsigned int x, unsigned int y);
			static unsigned int getIdByCoords(const Vector2i p);
			static Vector2i getTopLeftCorner(const unsigned int id); // Top-left and bottom-right
			static Vector2i getTopLeftCornerByCoords(const Vector2i point); // Top-left and bottom-right
			static const std::vector<unsigned int> getNeighbors(const std::vector<std::unique_ptr<AbstractSector>>& sectors, const Vector2i center);

		protected:
			std::vector<unsigned int> portalIds;
		};

		class Sector : public AbstractSector {
		public:
			using AbstractSector::AbstractSector;

			void setTile(const unsigned int x, const unsigned int y, Tile tile) override;
			Tile getTile(const unsigned int x, const unsigned int y) const override;
			Tile getTile(const Vector2i& p) const override;
			bool checkIsEmpty() const override;

		private:
			tileArrayT tiles;
		};

		// Empty sector - optimization. Functions in this sector should always return COST_MIN.
		class EmptySector : public AbstractSector {
		public:
			using AbstractSector::AbstractSector;

			void setTile(const unsigned int x, const unsigned int y, Tile tile) override;
			Tile getTile(const unsigned int x, const unsigned int y) const override;
			Tile getTile(const Vector2i& p) const override;
			bool checkIsEmpty() const override;
			bool isEmpty() const override;

		private:
			std::array<std::array<Tile, 0>, 0> tiles {};
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
			
			AbstractSector* firstSector = nullptr;
			AbstractSector* secondSector = nullptr;
			pointsT firstSectorPoints;
			pointsT secondSectorPoints;
			std::vector<unsigned int> neighbors;

			Portal() = default;
			Portal(const Portal&) = delete;
			Portal& operator=(const Portal&) = delete;
			Portal(Portal&&) = default;
			Portal& operator=(Portal&&) = default;
			~Portal() = default;

			Portal(AbstractSector* sector1, AbstractSector* sector2, pointsT& firstSectorPoints, pointsT& secondSectorPoints);

			bool isValid() const;
			Vector2i getFirstSectorCenter() const;
			Vector2i getSecondSectorCenter() const;
		};

		typedef std::vector<std::unique_ptr<AbstractSector>> sectorListT;
		// Map of [portalId, Portal]. Because portals can be removed, portal ids might be not continuous
		typedef std::map<unsigned int, Portal> portalMapT;

		// Generic A* algorithm. Reimplemented to better suit needs of flowfield.
		// The one in astar.cpp is absolutely NOT reusable, unfortunately.
		// PLEASE KEEP THIS CODE HERE CLEAN!
		class AbstractAStar {
		public:
			AbstractAStar() = delete;
			AbstractAStar(const AbstractAStar&) = delete;
			AbstractAStar& operator=(const AbstractAStar&) = delete;
			AbstractAStar(AbstractAStar &&) = delete;
			AbstractAStar &operator=(AbstractAStar &&) = delete;
			explicit AbstractAStar(unsigned int goal) : goal(goal) {};

			// Returns indexes of subsequent nodes in the path. Empty container means no path exists.
			const std::deque<unsigned int> findPath(const unsigned int startingIndex, const unsigned int nodes);
			virtual bool findPathExists(const unsigned int startingIndex, const unsigned int nodes);
		protected:
			// Implementation of following functions depends on data (tile, portal, ...)

			// Returns indexes of neighbors for given node
			virtual const std::vector<unsigned int> getNeighbors(unsigned int index) = 0;

			// Returns distance between current node and considered node.
			// For grid, return "10 * tile.cost" for vertical or horizontal, and "14 * tile.cost" for diagonal. This avoids costly `sqrt(2)` for grids.
			virtual unsigned int distance(unsigned int current, unsigned int neighbor) = 0;

			// The heuristic function. Returns expected cost of moving form `start` to `goal`. Use octal for diagonal and maybe Euclidean for portals
			// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
			virtual unsigned int heuristic(unsigned int start) = 0;

			virtual ~AbstractAStar() = default;

			unsigned int goal;
			///
		private:
			struct Node {
				unsigned int index;
				unsigned int cost;
				unsigned int heuristic;

				inline bool operator<(const Node& other) const {
					// We want the top element to have lowest cost
					return (cost + heuristic) > (other.cost + other.heuristic);
				}
			};

			// For each node, which node it can most efficiently be reached from
			std::map<unsigned int, unsigned int> cameFrom {};

			const std::deque<unsigned int> reconstructPath(unsigned int start);

			unsigned int _debugNodesVisited = 0;

			void logDebugNodesStats(unsigned int nodesTotal, int nodesInPath);
		};

		// This class works only in one given sector. `startingIndex` and `goal` therefore must be in range [0 ... SECTOR_SIZE^2)
		class TileAStar : public AbstractAStar {
		public:
			TileAStar(const unsigned goal, const sectorListT& sectors);

			bool findPathExists(const unsigned int startingIndex, const unsigned int nodes) override;

		protected:
			const std::vector<unsigned int> getNeighbors(const unsigned int index) override;
			unsigned int distance(const unsigned int current, const unsigned int neighbor) override;
			unsigned int heuristic(unsigned int start) override; // octile heuristic is preferred
		private:
			Vector2i goalPoint;
			const sectorListT& sectors;
			unsigned int sectorId;

			unsigned int distanceCommon(const Vector2i point1, const Vector2i point2, const unsigned int cost) const;
		};

		class PortalAStar : public AbstractAStar {
		public:
			PortalAStar(const unsigned int goal, portalMapT& portals)
				: AbstractAStar(goal), portals(portals), goalPortal(portals[goal]) {
			}

		protected:
			const std::vector<unsigned int> getNeighbors(unsigned int index) override;
			unsigned int distance(unsigned int current, unsigned int neighbor) override;
			unsigned int heuristic(unsigned int start) override; // straight-line (any angle) heuristic is preferred
		private:
			portalMapT& portals;
			const Portal& goalPortal;

			unsigned int distanceCommon(const Portal& portal1, const Portal& portal2) const;
		};

		// Defined in flowfield.cpp
		extern std::mutex logMutex;

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
						unsigned int took = timer.elapsed();
						debug(LOG_FLOWFIELD, "FuturedTask (%s) took %d ms", typeid(*this).name(), took);
					}
				} catch (const std::exception &ex) {
					promise.set_exception(std::current_exception());
					{
						std::lock_guard<std::mutex> lock(logMutex);
						debug(LOG_ERROR, "Exception in thread pool worker: %s", ex.what());
					}

					if (EXCEPTION_IN_THREADPOOL_SHOULD_TERMINATE) {
						std::terminate();
					}
				}
			}

			std::future<T> getFuture() {
				return promise.get_future();
			}
			~FuturedTask() = default;
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
			PathRequestTask(const Vector2i mapSource, const Vector2i mapGoal, const PROPULSION_TYPE propulsion)
				: mapSource(mapSource), mapGoal(mapGoal), propulsion(propulsion) {
			}
			void runPromised() override;
			
			~PathRequestTask() = default;
		private:
			const Vector2i mapSource;
			const Vector2i mapGoal;
			const PROPULSION_TYPE propulsion;

			unsigned int _debugCacheHits = 0;
			unsigned int _debugCacheMisses = 0;

			const std::vector<std::pair<unsigned int, unsigned int>> portalWalker(const unsigned int sourcePortalId, const unsigned int goalPortalId);
			std::vector<std::future<bool>> scheduleFlowFields(const std::vector<std::pair<unsigned int, unsigned int>>& path);
		};

		class FlowFieldSector final {
		public:
			struct VectorT {
				float x;
				float y;

				void normalize() {
					const float length = std::sqrt(std::pow(x, 2) + std::pow(y, 2));

					if (length != 0) {
						x /= length;
						y /= length;
					}
				}
			};
			typedef std::array<std::array<VectorT, SECTOR_SIZE>, SECTOR_SIZE> vectorArrayT;

			FlowFieldSector() = default;
			FlowFieldSector& operator=(FlowFieldSector&) = delete;
			FlowFieldSector& operator=(FlowFieldSector&&) = delete;
			FlowFieldSector(FlowFieldSector&) = delete;
			FlowFieldSector(FlowFieldSector&&) = default;
			~FlowFieldSector() = default;

			void setVector(const unsigned int x, const unsigned int y, VectorT vector);
			VectorT getVector(const unsigned int x, const unsigned int y) const;
			VectorT getVector(const Vector2i& p) const;

		private:
			vectorArrayT vectors;
		};

		// Promise states whether flow field calculation has completed for given sector
		class FlowfieldCalcTask : public FuturedTask<bool> {
		public:
			// Takes goals by copy. Need to control lifetime of goals myself (goals can be constructed ad-hoc)
			FlowfieldCalcTask(const Portal::pointsT goals, portalMapT& portals, const sectorListT& sectors, PROPULSION_TYPE propulsion);
			void runPromised() override;
			~FlowfieldCalcTask() = default;
		private:
			struct Node {
				unsigned short prececessorCost;
				unsigned int index;

				bool operator<(const Node& other) const {
					// We want top element to have lowest cost
					return prececessorCost > other.prececessorCost;
				}
			};

			Sector integrationField;
			FlowFieldSector flowField;
			
			// Constructor depends on member init order
			const Portal::pointsT goals;
			portalMapT& portals;
			const sectorListT& sectors;
			const unsigned int sectorId;
			const AbstractSector& sector;
			PROPULSION_TYPE propulsion;

			void calculateIntegrationField(const Portal::pointsT& points);
			void integratePoints(std::priority_queue<Node>& openSet);
			void calculateFlowField();
			float getCostOrExtrapolate(const short x, const short y, const unsigned short currentTileCost, const short elseX, const short elseY);
		};

		void initCostFields();
		void costFieldReplaceWithEmpty(sectorListT& sectors);
		void setupPortals();
		portalMapT setupPortalsForSectors(sectorListT& sectors);

		void destroyCostFields();
		void destroyPortals();
		void destroyFlowfieldCache();

		Tile createTile(const unsigned int x, const unsigned int y, PROPULSION_TYPE propulsion);

		/**
			* Portal detection, by axis.
			* axisStart - starting point for one of the axis
			* otherAxis1 - value from other axis, belonging to thisSector
			* otherAxis2 - value from other axis, belonging to otherSector
			* axisEndOut - last field checked for portal existence. Used as starting point in next iteration
			* If axisStart is "x", then "otherAxis" is y.
			*/
		Portal detectPortalByAxis(const unsigned int axisStart, const unsigned int axisEnd, const unsigned otherAxis1, const unsigned otherAxis2,
								  const bool isXAxis, AbstractSector& thisSector, AbstractSector& otherSector, unsigned int& axisEndOut);

		void connectPortals(portalMapT& portalMap, sectorListT& sectors);
		void connectPotentialNeighbor(std::pair<const unsigned int, Portal>& portalPair, const unsigned int potentialNeighbor,
									  TileAStar& pathfinder, portalMapT& portalMap, const bool isSectorEmpty);

		unsigned int pointToIndex(const Vector2i& p);
		Vector2i getPointByFlatIndex(unsigned int index);

		unsigned int straightLineDistance(const Vector2i source, const Vector2i destination);

		std::pair<unsigned int, unsigned int> mapSourceGoalToPortals(const Vector2i mapSource, const Vector2i mapGoal, const PROPULSION_TYPE propulsion);

		/*
			* Helps to decide if we should use firstSectorPoints or secondSectorPoints as goal points
			* See Portal class for more explanation
			*/
		bool isForward(const Vector2i& source, const Vector2i& firstSectorGoal, const Vector2i& secondSectorGoal);

		const std::vector<std::pair<unsigned int, unsigned int>> getPathFromCache(const unsigned int sourcePortalId,
			const unsigned int goalPortalId, const PROPULSION_TYPE propulsion);
		
		const Portal::pointsT portalToGoals(const Portal& portal, const Vector2i currentPosition);
		
		const Vector2f getMovementVector(const unsigned int nextPortalId, const Vector2i currentPosition, const PROPULSION_TYPE propulsion);
		const Vector2f getMovementVectorToFinalGoals(const Portal::pointsT goals, const Vector2i currentPosition, const PROPULSION_TYPE propulsion);
		const Vector2f getMovementVector(const Portal::pointsT goals, const Vector2i currentPosition, const PROPULSION_TYPE propulsion);
		
		unsigned int getCurrentSectorId(const Vector2i currentPosition);

		int sectorIdByTwoPortalIds(unsigned int firstPortalId, unsigned int secondPortalId, const PROPULSION_TYPE propulsion);

		std::vector<Vector2i> portalPathToCoordsPath(const std::vector<std::pair<unsigned int, unsigned int>>& path, const PROPULSION_TYPE propulsion);

		//////////////////////////////////////////////////////////////////////////////////////////
		// +- x axis tile debug draw. Smaller values = less tiles drawn. "7" somewhat fits the default window resolution
		constexpr const unsigned int DEBUG_DRAW_X_DELTA = 7;
		// +- y axis tile debug draw
		constexpr const unsigned int DEBUG_DRAW_Y_DELTA = 6;

		void debugDrawCoords();
		void debugDrawCostField();
		void debugTileDrawCost(AbstractSector& sector, const unsigned int x, const unsigned int y,
							   const unsigned int screenX, const unsigned int screenY);
		void debugDrawPortals();
		void debugDrawPortalPath();

		void debugDrawIntegrationField();
		void debugDrawFlowField();
}
}