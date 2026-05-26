/* ane_private.h — Forward declarations for the private
 * AppleNeuralEngine.framework classes we dispatch into.
 *
 * Everything here is loaded via NSClassFromString + objc_msgSend; this
 * header documents the selectors we rely on. No symbols are linked
 * directly — the framework is dlopen'd at startup.
 */
#ifndef ANE_PRIVATE_H
#define ANE_PRIVATE_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/* The five private classes we look up by name. */
extern Class g_AneDescriptorCls;   /* _ANEInMemoryModelDescriptor */
extern Class g_AneInMemoryCls;     /* _ANEInMemoryModel */
extern Class g_AneRequestCls;      /* _ANERequest */
extern Class g_AneIOSurfaceCls;    /* _ANEIOSurfaceObject */
extern Class g_AneClientCls;       /* _ANEClient */

/* Selectors used (documentation only — invoked via objc_msgSend casts). */
/*  +[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]
 *  +[_ANEInMemoryModel inMemoryModelWithDescriptor:]
 *  -[_ANEInMemoryModel hexStringIdentifier]
 *  -[_ANEInMemoryModel compileWithQoS:options:error:]
 *  -[_ANEInMemoryModel loadWithQoS:options:error:]
 *  -[_ANEInMemoryModel unloadWithQoS:error:]
 *  +[_ANEIOSurfaceObject objectWithIOSurface:]
 *  +[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:
 *                weightsBuffer:perfStats:procedureIndex:]
 *  -[_ANEClient initWithRestrictedAccessAllowed:]
 *  -[_ANEClient doEvaluateDirectWithModel:options:request:qos:error:]
 */

/* Idempotent loader. Returns YES on success, NO if the framework or
 * any of the required classes cannot be found. */
BOOL ane_private_load(void);

/* Install the getUUID stub on _ANEInMemoryModel. Idempotent. */
void ane_private_install_uuid_stub(void);

#endif /* ANE_PRIVATE_H */
