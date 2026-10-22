import sys
from streamlit.runtime.state.session_state import SessionStateStatProvider

# Patch for Python 3.10.0rc1 typing.Protocol dataclass __init__ bug
def patch_session_state():
    def custom_init(self, session_mgr):
        self._session_mgr = session_mgr
    SessionStateStatProvider.__init__ = custom_init

patch_session_state()

from streamlit.web import cli as stcli

if __name__ == '__main__':
    # Pass arguments properly to streamlit run app.py
    sys.argv = ["streamlit", "run", "app.py"] + sys.argv[1:]
    sys.exit(stcli.main())
