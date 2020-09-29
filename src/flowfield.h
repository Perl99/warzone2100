#pragma once

#ifdef _MSC_VER
// Nameless struct in glm::vec2<>
#pragma warning(disable: 4201)
#endif // _MSC_VER

#include "fpath.h"
#include "movedef.h"
#include "lib/framework/frame.h" // for statsdef.h
#include "statsdef.h"
#include <deque>

/*
	Concept: http://www.gameaipro.com/GameAIPro/GameAIPro_Chapter23_Crowd_Pathfinding_and_Steering_Using_Flow_Field_Tiles.pdf
*/

/////////////////////////////////////////////////////////////////////

// TODO: avoiding tiles marked as "threat" (only AI, or anyone? It would be nice if your own droids would prefer to avoid enemy when retreating)
// TODO: maybe prefer visible tiles, or just discovered tiles (for player)
// Both things would go into integration field probably. Note that adding visibility stuff would quickly require most integration and flow fields to be thrown away, since visibility changes all the time.

// TODO: support marking sectors as dirty: portals needs to be rebuild, cost fields recalculated, and integration and flow fields for that sector must be removed from cache.

namespace flowfield
{
// TODO: disable before merge with main branch
#define I_AM_FLOW_FIELD_MAINTAINER_AND_WANT_TO_HAVE_MY_VIEW_OBSCURED 1

#if defined(NDEBUG) || !defined(I_AM_FLOW_FIELD_MAINTAINER_AND_WANT_TO_HAVE_MY_VIEW_OBSCURED)
	// General debug flag for things not related to other debug flags declared below
	constexpr const bool DEBUG_BUILD = false;

	constexpr const bool COST_FIELD_DEBUG = false;
	constexpr const bool INTEGRATION_FIELD_DEBUG = false;
	constexpr const bool VECTOR_FIELD_DEBUG = false;
	constexpr const bool PORTALS_DEBUG = false;
	constexpr const bool PORTAL_PATH_DEBUG = false;
	constexpr const bool DEBUG_A_STAR = false;
	constexpr const bool DEBUG_THREAD_POOL = false;
#else
	// General debug flag for things not related to other debug flags declared below
	constexpr const bool DEBUG_BUILD = true;

	// FEEL FREE TO DISABLE THOSE AS THEY OBSCURE THE VIEW
	constexpr const bool COST_FIELD_DEBUG = true;
	constexpr const bool INTEGRATION_FIELD_DEBUG = true;
	constexpr const bool VECTOR_FIELD_DEBUG = true;
	constexpr const bool PORTALS_DEBUG = true;
	constexpr const bool PORTAL_PATH_DEBUG = true;
	constexpr const bool DEBUG_A_STAR = true;
	constexpr const bool DEBUG_THREAD_POOL = true;
#endif

	void enable();
	bool isEnabled();

	void init();
	void destroy();

	/**
	 * Public interface same as fpathRoute for compatibility.
	 * Most fields are not yet used in current flowfield implementation, but I wanted it to be compatible with fpath.cpp
	 */
	void calculateFlowFieldsAsync(MOVE_CONTROL *psMove, unsigned id, int startX, int startY, int tX, int tY, PROPULSION_TYPE propulsionType,
								  DROID_TYPE droidType, FPATH_MOVETYPE moveType, int owner, bool acceptNearest, StructureBounds const &dstStructure);

	/**
	 * Path consisting of pairs (sectorId, portalId)
	 */
	std::vector<std::pair<unsigned int, unsigned int>> getPathFromCache(unsigned startX, unsigned startY, unsigned tX, unsigned tY,
		PROPULSION_TYPE propulsion);

	Vector2f getMovementVector(unsigned int nextPortalId, unsigned currentX, unsigned currentY, PROPULSION_TYPE propulsion);
	Vector2f getMovementVectorToFinalGoal(unsigned tX, unsigned tY, unsigned currentX, unsigned currentY, PROPULSION_TYPE propulsion);

	unsigned int getCurrentSectorId(unsigned currentX, unsigned currentY);

	std::vector<Vector2i> portalPathToCoordsPath(const std::vector<std::pair<unsigned, unsigned>>& path, PROPULSION_TYPE propulsion);

	void debugDraw();
}
