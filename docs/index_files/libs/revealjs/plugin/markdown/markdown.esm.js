function e(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t &&
      (r = r.filter(function (t) {
        return Object.getOwnPropertyDescriptor(e, t).enumerable;
      })),
      n.push.apply(n, r);
  }
  return n;
}
function t(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function n(e, t) {
  for (var n = 0; n < t.length; n++) {
    var r = t[n];
    (r.enumerable = r.enumerable || !1),
      (r.configurable = !0),
      "value" in r && (r.writable = !0),
      Object.defineProperty(e, r.key, r);
  }
}
function r(e, t, r) {
  return t && n(e.prototype, t), r && n(e, r), e;
}
function u(e, t, n) {
  return (
    t in e
      ? Object.defineProperty(e, t, {
          value: n,
          enumerable: !0,
          configurable: !0,
          writable: !0,
        })
      : (e[t] = n),
    e
  );
}
function i(e, t) {
  if (null == e) return {};
  var n,
    r,
    u = (function (e, t) {
      if (null == e) return {};
      var n,
        r,
        u = {},
        i = Object.keys(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]), t.indexOf(n) >= 0 || (u[n] = e[n]);
      return u;
    })(e, t);
  if (Object.getOwnPropertySymbols) {
    var i = Object.getOwnPropertySymbols(e);
    for (r = 0; r < i.length; r++)
      (n = i[r]),
        t.indexOf(n) >= 0 ||
          (Object.prototype.propertyIsEnumerable.call(e, n) && (u[n] = e[n]));
  }
  return u;
}
function o(e, t) {
  return (
    (function (e) {
      if (Array.isArray(e)) return e;
    })(e) ||
    (function (e, t) {
      var n =
        e &&
        (("undefined" != typeof Symbol && e[Symbol.iterator]) ||
          e["@@iterator"]);
      if (null == n) return;
      var r,
        u,
        i = [],
        o = !0,
        a = !1;
      try {
        for (
          n = n.call(e);
          !(o = (r = n.next()).done) && (i.push(r.value), !t || i.length !== t);
          o = !0
        );
      } catch (e) {
        (a = !0), (u = e);
      } finally {
        try {
          o || null == n.return || n.return();
        } finally {
          if (a) throw u;
        }
      }
      return i;
    })(e, t) ||
    a(e, t) ||
    (function () {
      throw new TypeError(
        "Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
      );
    })()
  );
}
function a(e, t) {
  if (e) {
    if ("string" == typeof e) return s(e, t);
    var n = Object.prototype.toString.call(e).slice(8, -1);
    return (
      "Object" === n && e.constructor && (n = e.constructor.name),
      "Map" === n || "Set" === n
        ? Array.from(e)
        : "Arguments" === n ||
            /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
          ? s(e, t)
          : void 0
    );
  }
}
function s(e, t) {
  (null == t || t > e.length) && (t = e.length);
  for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
  return r;
}
function l(e, t) {
  var n =
    ("undefined" != typeof Symbol && e[Symbol.iterator]) || e["@@iterator"];
  if (!n) {
    if (
      Array.isArray(e) ||
      (n = a(e)) ||
      (t && e && "number" == typeof e.length)
    ) {
      n && (e = n);
      var r = 0,
        u = function () {};
      return {
        s: u,
        n: function () {
          return r >= e.length ? { done: !0 } : { done: !1, value: e[r++] };
        },
        e: function (e) {
          throw e;
        },
        f: u,
      };
    }
    throw new TypeError(
      "Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
    );
  }
  var i,
    o = !0,
    s = !1;
  return {
    s: function () {
      n = n.call(e);
    },
    n: function () {
      var e = n.next();
      return (o = e.done), e;
    },
    e: function (e) {
      (s = !0), (i = e);
    },
    f: function () {
      try {
        o || null == n.return || n.return();
      } finally {
        if (s) throw i;
      }
    },
  };
}
var c =
    "undefined" != typeof globalThis
      ? globalThis
      : "undefined" != typeof window
        ? window
        : "undefined" != typeof global
          ? global
          : "undefined" != typeof self
            ? self
            : {},
  f = function (e) {
    return e && e.Math == Math && e;
  },
  p =
    f("object" == typeof globalThis && globalThis) ||
    f("object" == typeof window && window) ||
    f("object" == typeof self && self) ||
    f("object" == typeof c && c) ||
    (function () {
      return this;
    })() ||
    Function("return this")(),
  h = {},
  D = function (e) {
    try {
      return !!e();
    } catch (e) {
      return !0;
    }
  },
  g = !D(function () {
    return (
      7 !=
      Object.defineProperty({}, 1, {
        get: function () {
          return 7;
        },
      })[1]
    );
  }),
  d = {},
  v = {}.propertyIsEnumerable,
  y = Object.getOwnPropertyDescriptor,
  A = y && !v.call({ 1: 2 }, 1);
d.f = A
  ? function (e) {
      var t = y(this, e);
      return !!t && t.enumerable;
    }
  : v;
var m = function (e, t) {
    return {
      enumerable: !(1 & e),
      configurable: !(2 & e),
      writable: !(4 & e),
      value: t,
    };
  },
  k = {}.toString,
  E = function (e) {
    return k.call(e).slice(8, -1);
  },
  x = E,
  F = "".split,
  b = D(function () {
    return !Object("z").propertyIsEnumerable(0);
  })
    ? function (e) {
        return "String" == x(e) ? F.call(e, "") : Object(e);
      }
    : Object,
  C = function (e) {
    if (null == e) throw TypeError("Can't call method on " + e);
    return e;
  },
  w = b,
  B = C,
  S = function (e) {
    return w(B(e));
  },
  _ = function (e) {
    return "object" == typeof e ? null !== e : "function" == typeof e;
  },
  T = _,
  O = function (e, t) {
    if (!T(e)) return e;
    var n, r;
    if (t && "function" == typeof (n = e.toString) && !T((r = n.call(e))))
      return r;
    if ("function" == typeof (n = e.valueOf) && !T((r = n.call(e)))) return r;
    if (!t && "function" == typeof (n = e.toString) && !T((r = n.call(e))))
      return r;
    throw TypeError("Can't convert object to primitive value");
  },
  R = C,
  I = function (e) {
    return Object(R(e));
  },
  j = I,
  z = {}.hasOwnProperty,
  $ = function (e, t) {
    return z.call(j(e), t);
  },
  P = _,
  L = p.document,
  M = P(L) && P(L.createElement),
  N = function (e) {
    return M ? L.createElement(e) : {};
  },
  U = N,
  q =
    !g &&
    !D(function () {
      return (
        7 !=
        Object.defineProperty(U("div"), "a", {
          get: function () {
            return 7;
          },
        }).a
      );
    }),
  Z = g,
  G = d,
  H = m,
  Q = S,
  V = O,
  Y = $,
  K = q,
  X = Object.getOwnPropertyDescriptor;
h.f = Z
  ? X
  : function (e, t) {
      if (((e = Q(e)), (t = V(t, !0)), K))
        try {
          return X(e, t);
        } catch (e) {}
      if (Y(e, t)) return H(!G.f.call(e, t), e[t]);
    };
var W = {},
  J = _,
  ee = function (e) {
    if (!J(e)) throw TypeError(String(e) + " is not an object");
    return e;
  },
  te = g,
  ne = q,
  re = ee,
  ue = O,
  ie = Object.defineProperty;
W.f = te
  ? ie
  : function (e, t, n) {
      if ((re(e), (t = ue(t, !0)), re(n), ne))
        try {
          return ie(e, t, n);
        } catch (e) {}
      if ("get" in n || "set" in n) throw TypeError("Accessors not supported");
      return "value" in n && (e[t] = n.value), e;
    };
var oe = W,
  ae = m,
  se = g
    ? function (e, t, n) {
        return oe.f(e, t, ae(1, n));
      }
    : function (e, t, n) {
        return (e[t] = n), e;
      },
  le = { exports: {} },
  ce = p,
  fe = se,
  pe = function (e, t) {
    try {
      fe(ce, e, t);
    } catch (n) {
      ce[e] = t;
    }
    return t;
  },
  he = pe,
  De = p["__core-js_shared__"] || he("__core-js_shared__", {}),
  ge = De,
  de = Function.toString;
"function" != typeof ge.inspectSource &&
  (ge.inspectSource = function (e) {
    return de.call(e);
  });
var ve = ge.inspectSource,
  ye = ve,
  Ae = p.WeakMap,
  me = "function" == typeof Ae && /native code/.test(ye(Ae)),
  ke = { exports: {} },
  Ee = De;
(ke.exports = function (e, t) {
  return Ee[e] || (Ee[e] = void 0 !== t ? t : {});
})("versions", []).push({
  version: "3.12.1",
  mode: "global",
  copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
});
var xe,
  Fe,
  be,
  Ce = 0,
  we = Math.random(),
  Be = function (e) {
    return (
      "Symbol(" +
      String(void 0 === e ? "" : e) +
      ")_" +
      (++Ce + we).toString(36)
    );
  },
  Se = ke.exports,
  _e = Be,
  Te = Se("keys"),
  Oe = function (e) {
    return Te[e] || (Te[e] = _e(e));
  },
  Re = {},
  Ie = me,
  je = _,
  ze = se,
  $e = $,
  Pe = De,
  Le = Oe,
  Me = Re,
  Ne = p.WeakMap;
if (Ie || Pe.state) {
  var Ue = Pe.state || (Pe.state = new Ne()),
    qe = Ue.get,
    Ze = Ue.has,
    Ge = Ue.set;
  (xe = function (e, t) {
    if (Ze.call(Ue, e)) throw new TypeError("Object already initialized");
    return (t.facade = e), Ge.call(Ue, e, t), t;
  }),
    (Fe = function (e) {
      return qe.call(Ue, e) || {};
    }),
    (be = function (e) {
      return Ze.call(Ue, e);
    });
} else {
  var He = Le("state");
  (Me[He] = !0),
    (xe = function (e, t) {
      if ($e(e, He)) throw new TypeError("Object already initialized");
      return (t.facade = e), ze(e, He, t), t;
    }),
    (Fe = function (e) {
      return $e(e, He) ? e[He] : {};
    }),
    (be = function (e) {
      return $e(e, He);
    });
}
var Qe = {
    set: xe,
    get: Fe,
    has: be,
    enforce: function (e) {
      return be(e) ? Fe(e) : xe(e, {});
    },
    getterFor: function (e) {
      return function (t) {
        var n;
        if (!je(t) || (n = Fe(t)).type !== e)
          throw TypeError("Incompatible receiver, " + e + " required");
        return n;
      };
    },
  },
  Ve = p,
  Ye = se,
  Ke = $,
  Xe = pe,
  We = ve,
  Je = Qe.get,
  et = Qe.enforce,
  tt = String(String).split("String");
(le.exports = function (e, t, n, r) {
  var u,
    i = !!r && !!r.unsafe,
    o = !!r && !!r.enumerable,
    a = !!r && !!r.noTargetGet;
  "function" == typeof n &&
    ("string" != typeof t || Ke(n, "name") || Ye(n, "name", t),
    (u = et(n)).source || (u.source = tt.join("string" == typeof t ? t : ""))),
    e !== Ve
      ? (i ? !a && e[t] && (o = !0) : delete e[t], o ? (e[t] = n) : Ye(e, t, n))
      : o
        ? (e[t] = n)
        : Xe(t, n);
})(Function.prototype, "toString", function () {
  return ("function" == typeof this && Je(this).source) || We(this);
});
var nt = p,
  rt = p,
  ut = function (e) {
    return "function" == typeof e ? e : void 0;
  },
  it = function (e, t) {
    return arguments.length < 2
      ? ut(nt[e]) || ut(rt[e])
      : (nt[e] && nt[e][t]) || (rt[e] && rt[e][t]);
  },
  ot = {},
  at = Math.ceil,
  st = Math.floor,
  lt = function (e) {
    return isNaN((e = +e)) ? 0 : (e > 0 ? st : at)(e);
  },
  ct = lt,
  ft = Math.min,
  pt = function (e) {
    return e > 0 ? ft(ct(e), 9007199254740991) : 0;
  },
  ht = lt,
  Dt = Math.max,
  gt = Math.min,
  dt = function (e, t) {
    var n = ht(e);
    return n < 0 ? Dt(n + t, 0) : gt(n, t);
  },
  vt = S,
  yt = pt,
  At = dt,
  mt = function (e) {
    return function (t, n, r) {
      var u,
        i = vt(t),
        o = yt(i.length),
        a = At(r, o);
      if (e && n != n) {
        for (; o > a; ) if ((u = i[a++]) != u) return !0;
      } else
        for (; o > a; a++) if ((e || a in i) && i[a] === n) return e || a || 0;
      return !e && -1;
    };
  },
  kt = { includes: mt(!0), indexOf: mt(!1) },
  Et = $,
  xt = S,
  Ft = kt.indexOf,
  bt = Re,
  Ct = function (e, t) {
    var n,
      r = xt(e),
      u = 0,
      i = [];
    for (n in r) !Et(bt, n) && Et(r, n) && i.push(n);
    for (; t.length > u; ) Et(r, (n = t[u++])) && (~Ft(i, n) || i.push(n));
    return i;
  },
  wt = [
    "constructor",
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toLocaleString",
    "toString",
    "valueOf",
  ],
  Bt = Ct,
  St = wt.concat("length", "prototype");
ot.f =
  Object.getOwnPropertyNames ||
  function (e) {
    return Bt(e, St);
  };
var _t = {};
_t.f = Object.getOwnPropertySymbols;
var Tt = ot,
  Ot = _t,
  Rt = ee,
  It =
    it("Reflect", "ownKeys") ||
    function (e) {
      var t = Tt.f(Rt(e)),
        n = Ot.f;
      return n ? t.concat(n(e)) : t;
    },
  jt = $,
  zt = It,
  $t = h,
  Pt = W,
  Lt = D,
  Mt = /#|\.prototype\./,
  Nt = function (e, t) {
    var n = qt[Ut(e)];
    return n == Gt || (n != Zt && ("function" == typeof t ? Lt(t) : !!t));
  },
  Ut = (Nt.normalize = function (e) {
    return String(e).replace(Mt, ".").toLowerCase();
  }),
  qt = (Nt.data = {}),
  Zt = (Nt.NATIVE = "N"),
  Gt = (Nt.POLYFILL = "P"),
  Ht = Nt,
  Qt = p,
  Vt = h.f,
  Yt = se,
  Kt = le.exports,
  Xt = pe,
  Wt = function (e, t) {
    for (var n = zt(t), r = Pt.f, u = $t.f, i = 0; i < n.length; i++) {
      var o = n[i];
      jt(e, o) || r(e, o, u(t, o));
    }
  },
  Jt = Ht,
  en = function (e, t) {
    var n,
      r,
      u,
      i,
      o,
      a = e.target,
      s = e.global,
      l = e.stat;
    if ((n = s ? Qt : l ? Qt[a] || Xt(a, {}) : (Qt[a] || {}).prototype))
      for (r in t) {
        if (
          ((i = t[r]),
          (u = e.noTargetGet ? (o = Vt(n, r)) && o.value : n[r]),
          !Jt(s ? r : a + (l ? "." : "#") + r, e.forced) && void 0 !== u)
        ) {
          if (typeof i == typeof u) continue;
          Wt(i, u);
        }
        (e.sham || (u && u.sham)) && Yt(i, "sham", !0), Kt(n, r, i, e);
      }
  },
  tn = ee,
  nn = function () {
    var e = tn(this),
      t = "";
    return (
      e.global && (t += "g"),
      e.ignoreCase && (t += "i"),
      e.multiline && (t += "m"),
      e.dotAll && (t += "s"),
      e.unicode && (t += "u"),
      e.sticky && (t += "y"),
      t
    );
  },
  rn = {},
  un = D;
function on(e, t) {
  return RegExp(e, t);
}
(rn.UNSUPPORTED_Y = un(function () {
  var e = on("a", "y");
  return (e.lastIndex = 2), null != e.exec("abcd");
})),
  (rn.BROKEN_CARET = un(function () {
    var e = on("^r", "gy");
    return (e.lastIndex = 2), null != e.exec("str");
  }));
var an = nn,
  sn = rn,
  ln = ke.exports,
  cn = RegExp.prototype.exec,
  fn = ln("native-string-replace", String.prototype.replace),
  pn = cn,
  hn = (function () {
    var e = /a/,
      t = /b*/g;
    return (
      cn.call(e, "a"), cn.call(t, "a"), 0 !== e.lastIndex || 0 !== t.lastIndex
    );
  })(),
  Dn = sn.UNSUPPORTED_Y || sn.BROKEN_CARET,
  gn = void 0 !== /()??/.exec("")[1];
(hn || gn || Dn) &&
  (pn = function (e) {
    var t,
      n,
      r,
      u,
      i = this,
      o = Dn && i.sticky,
      a = an.call(i),
      s = i.source,
      l = 0,
      c = e;
    return (
      o &&
        (-1 === (a = a.replace("y", "")).indexOf("g") && (a += "g"),
        (c = String(e).slice(i.lastIndex)),
        i.lastIndex > 0 &&
          (!i.multiline || (i.multiline && "\n" !== e[i.lastIndex - 1])) &&
          ((s = "(?: " + s + ")"), (c = " " + c), l++),
        (n = new RegExp("^(?:" + s + ")", a))),
      gn && (n = new RegExp("^" + s + "$(?!\\s)", a)),
      hn && (t = i.lastIndex),
      (r = cn.call(o ? n : i, c)),
      o
        ? r
          ? ((r.input = r.input.slice(l)),
            (r[0] = r[0].slice(l)),
            (r.index = i.lastIndex),
            (i.lastIndex += r[0].length))
          : (i.lastIndex = 0)
        : hn && r && (i.lastIndex = i.global ? r.index + r[0].length : t),
      gn &&
        r &&
        r.length > 1 &&
        fn.call(r[0], n, function () {
          for (u = 1; u < arguments.length - 2; u++)
            void 0 === arguments[u] && (r[u] = void 0);
        }),
      r
    );
  });
var dn = pn;
en({ target: "RegExp", proto: !0, forced: /./.exec !== dn }, { exec: dn });
var vn,
  yn,
  An = it("navigator", "userAgent") || "",
  mn = An,
  kn = p.process,
  En = kn && kn.versions,
  xn = En && En.v8;
xn
  ? (yn = (vn = xn.split("."))[0] < 4 ? 1 : vn[0] + vn[1])
  : mn &&
    (!(vn = mn.match(/Edge\/(\d+)/)) || vn[1] >= 74) &&
    (vn = mn.match(/Chrome\/(\d+)/)) &&
    (yn = vn[1]);
var Fn = yn && +yn,
  bn = Fn,
  Cn = D,
  wn =
    !!Object.getOwnPropertySymbols &&
    !Cn(function () {
      return !String(Symbol()) || (!Symbol.sham && bn && bn < 41);
    }),
  Bn = wn && !Symbol.sham && "symbol" == typeof Symbol.iterator,
  Sn = p,
  _n = ke.exports,
  Tn = $,
  On = Be,
  Rn = wn,
  In = Bn,
  jn = _n("wks"),
  zn = Sn.Symbol,
  $n = In ? zn : (zn && zn.withoutSetter) || On,
  Pn = function (e) {
    return (
      (Tn(jn, e) && (Rn || "string" == typeof jn[e])) ||
        (Rn && Tn(zn, e) ? (jn[e] = zn[e]) : (jn[e] = $n("Symbol." + e))),
      jn[e]
    );
  },
  Ln = le.exports,
  Mn = dn,
  Nn = D,
  Un = Pn,
  qn = se,
  Zn = Un("species"),
  Gn = RegExp.prototype,
  Hn = !Nn(function () {
    var e = /./;
    return (
      (e.exec = function () {
        var e = [];
        return (e.groups = { a: "7" }), e;
      }),
      "7" !== "".replace(e, "$<a>")
    );
  }),
  Qn = "$0" === "a".replace(/./, "$0"),
  Vn = Un("replace"),
  Yn = !!/./[Vn] && "" === /./[Vn]("a", "$0"),
  Kn = !Nn(function () {
    var e = /(?:)/,
      t = e.exec;
    e.exec = function () {
      return t.apply(this, arguments);
    };
    var n = "ab".split(e);
    return 2 !== n.length || "a" !== n[0] || "b" !== n[1];
  }),
  Xn = function (e, t, n, r) {
    var u = Un(e),
      i = !Nn(function () {
        var t = {};
        return (
          (t[u] = function () {
            return 7;
          }),
          7 != ""[e](t)
        );
      }),
      o =
        i &&
        !Nn(function () {
          var t = !1,
            n = /a/;
          return (
            "split" === e &&
              (((n = {}).constructor = {}),
              (n.constructor[Zn] = function () {
                return n;
              }),
              (n.flags = ""),
              (n[u] = /./[u])),
            (n.exec = function () {
              return (t = !0), null;
            }),
            n[u](""),
            !t
          );
        });
    if (
      !i ||
      !o ||
      ("replace" === e && (!Hn || !Qn || Yn)) ||
      ("split" === e && !Kn)
    ) {
      var a = /./[u],
        s = n(
          u,
          ""[e],
          function (e, t, n, r, u) {
            var o = t.exec;
            return o === Mn || o === Gn.exec
              ? i && !u
                ? { done: !0, value: a.call(t, n, r) }
                : { done: !0, value: e.call(n, t, r) }
              : { done: !1 };
          },
          {
            REPLACE_KEEPS_$0: Qn,
            REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE: Yn,
          },
        ),
        l = s[0],
        c = s[1];
      Ln(String.prototype, e, l),
        Ln(
          Gn,
          u,
          2 == t
            ? function (e, t) {
                return c.call(e, this, t);
              }
            : function (e) {
                return c.call(e, this);
              },
        );
    }
    r && qn(Gn[u], "sham", !0);
  },
  Wn = lt,
  Jn = C,
  er = function (e) {
    return function (t, n) {
      var r,
        u,
        i = String(Jn(t)),
        o = Wn(n),
        a = i.length;
      return o < 0 || o >= a
        ? e
          ? ""
          : void 0
        : (r = i.charCodeAt(o)) < 55296 ||
            r > 56319 ||
            o + 1 === a ||
            (u = i.charCodeAt(o + 1)) < 56320 ||
            u > 57343
          ? e
            ? i.charAt(o)
            : r
          : e
            ? i.slice(o, o + 2)
            : u - 56320 + ((r - 55296) << 10) + 65536;
    };
  },
  tr = { codeAt: er(!1), charAt: er(!0) },
  nr = tr.charAt,
  rr = function (e, t, n) {
    return t + (n ? nr(e, t).length : 1);
  },
  ur = I,
  ir = Math.floor,
  or = "".replace,
  ar = /\$([$&'`]|\d{1,2}|<[^>]*>)/g,
  sr = /\$([$&'`]|\d{1,2})/g,
  lr = E,
  cr = dn,
  fr = function (e, t) {
    var n = e.exec;
    if ("function" == typeof n) {
      var r = n.call(e, t);
      if ("object" != typeof r)
        throw TypeError(
          "RegExp exec method returned something other than an Object or null",
        );
      return r;
    }
    if ("RegExp" !== lr(e))
      throw TypeError("RegExp#exec called on incompatible receiver");
    return cr.call(e, t);
  },
  pr = Xn,
  hr = ee,
  Dr = pt,
  gr = lt,
  dr = C,
  vr = rr,
  yr = function (e, t, n, r, u, i) {
    var o = n + e.length,
      a = r.length,
      s = sr;
    return (
      void 0 !== u && ((u = ur(u)), (s = ar)),
      or.call(i, s, function (i, s) {
        var l;
        switch (s.charAt(0)) {
          case "$":
            return "$";
          case "&":
            return e;
          case "`":
            return t.slice(0, n);
          case "'":
            return t.slice(o);
          case "<":
            l = u[s.slice(1, -1)];
            break;
          default:
            var c = +s;
            if (0 === c) return i;
            if (c > a) {
              var f = ir(c / 10);
              return 0 === f
                ? i
                : f <= a
                  ? void 0 === r[f - 1]
                    ? s.charAt(1)
                    : r[f - 1] + s.charAt(1)
                  : i;
            }
            l = r[c - 1];
        }
        return void 0 === l ? "" : l;
      })
    );
  },
  Ar = fr,
  mr = Math.max,
  kr = Math.min;
pr("replace", 2, function (e, t, n, r) {
  var u = r.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,
    i = r.REPLACE_KEEPS_$0,
    o = u ? "$" : "$0";
  return [
    function (n, r) {
      var u = dr(this),
        i = null == n ? void 0 : n[e];
      return void 0 !== i ? i.call(n, u, r) : t.call(String(u), n, r);
    },
    function (e, r) {
      if ((!u && i) || ("string" == typeof r && -1 === r.indexOf(o))) {
        var a = n(t, e, this, r);
        if (a.done) return a.value;
      }
      var s = hr(e),
        l = String(this),
        c = "function" == typeof r;
      c || (r = String(r));
      var f = s.global;
      if (f) {
        var p = s.unicode;
        s.lastIndex = 0;
      }
      for (var h = []; ; ) {
        var D = Ar(s, l);
        if (null === D) break;
        if ((h.push(D), !f)) break;
        "" === String(D[0]) && (s.lastIndex = vr(l, Dr(s.lastIndex), p));
      }
      for (var g, d = "", v = 0, y = 0; y < h.length; y++) {
        D = h[y];
        for (
          var A = String(D[0]),
            m = mr(kr(gr(D.index), l.length), 0),
            k = [],
            E = 1;
          E < D.length;
          E++
        )
          k.push(void 0 === (g = D[E]) ? g : String(g));
        var x = D.groups;
        if (c) {
          var F = [A].concat(k, m, l);
          void 0 !== x && F.push(x);
          var b = String(r.apply(void 0, F));
        } else b = yr(A, l, m, k, x, r);
        m >= v && ((d += l.slice(v, m) + b), (v = m + A.length));
      }
      return d + l.slice(v);
    },
  ];
});
var Er = _,
  xr = ee,
  Fr = function (e) {
    if (!Er(e) && null !== e)
      throw TypeError("Can't set " + String(e) + " as a prototype");
    return e;
  },
  br =
    Object.setPrototypeOf ||
    ("__proto__" in {}
      ? (function () {
          var e,
            t = !1,
            n = {};
          try {
            (e = Object.getOwnPropertyDescriptor(
              Object.prototype,
              "__proto__",
            ).set).call(n, []),
              (t = n instanceof Array);
          } catch (e) {}
          return function (n, r) {
            return xr(n), Fr(r), t ? e.call(n, r) : (n.__proto__ = r), n;
          };
        })()
      : void 0),
  Cr = _,
  wr = br,
  Br = _,
  Sr = E,
  _r = Pn("match"),
  Tr = function (e) {
    var t;
    return Br(e) && (void 0 !== (t = e[_r]) ? !!t : "RegExp" == Sr(e));
  },
  Or = it,
  Rr = W,
  Ir = g,
  jr = Pn("species"),
  zr = function (e) {
    var t = Or(e),
      n = Rr.f;
    Ir &&
      t &&
      !t[jr] &&
      n(t, jr, {
        configurable: !0,
        get: function () {
          return this;
        },
      });
  },
  $r = g,
  Pr = p,
  Lr = Ht,
  Mr = function (e, t, n) {
    var r, u;
    return (
      wr &&
        "function" == typeof (r = t.constructor) &&
        r !== n &&
        Cr((u = r.prototype)) &&
        u !== n.prototype &&
        wr(e, u),
      e
    );
  },
  Nr = W.f,
  Ur = ot.f,
  qr = Tr,
  Zr = nn,
  Gr = rn,
  Hr = le.exports,
  Qr = D,
  Vr = Qe.enforce,
  Yr = zr,
  Kr = Pn("match"),
  Xr = Pr.RegExp,
  Wr = Xr.prototype,
  Jr = /a/g,
  eu = /a/g,
  tu = new Xr(Jr) !== Jr,
  nu = Gr.UNSUPPORTED_Y;
if (
  $r &&
  Lr(
    "RegExp",
    !tu ||
      nu ||
      Qr(function () {
        return (
          (eu[Kr] = !1), Xr(Jr) != Jr || Xr(eu) == eu || "/a/i" != Xr(Jr, "i")
        );
      }),
  )
) {
  for (
    var ru = function (e, t) {
        var n,
          r = this instanceof ru,
          u = qr(e),
          i = void 0 === t;
        if (!r && u && e.constructor === ru && i) return e;
        tu
          ? u && !i && (e = e.source)
          : e instanceof ru && (i && (t = Zr.call(e)), (e = e.source)),
          nu && (n = !!t && t.indexOf("y") > -1) && (t = t.replace(/y/g, ""));
        var o = Mr(tu ? new Xr(e, t) : Xr(e, t), r ? this : Wr, ru);
        nu && n && (Vr(o).sticky = !0);
        return o;
      },
      uu = function (e) {
        (e in ru) ||
          Nr(ru, e, {
            configurable: !0,
            get: function () {
              return Xr[e];
            },
            set: function (t) {
              Xr[e] = t;
            },
          });
      },
      iu = Ur(Xr),
      ou = 0;
    iu.length > ou;

  )
    uu(iu[ou++]);
  (Wr.constructor = ru), (ru.prototype = Wr), Hr(Pr, "RegExp", ru);
}
Yr("RegExp");
var au = le.exports,
  su = ee,
  lu = D,
  cu = nn,
  fu = RegExp.prototype,
  pu = fu.toString,
  hu = lu(function () {
    return "/a/b" != pu.call({ source: "a", flags: "b" });
  }),
  Du = "toString" != pu.name;
(hu || Du) &&
  au(
    RegExp.prototype,
    "toString",
    function () {
      var e = su(this),
        t = String(e.source),
        n = e.flags;
      return (
        "/" +
        t +
        "/" +
        String(
          void 0 === n && e instanceof RegExp && !("flags" in fu)
            ? cu.call(e)
            : n,
        )
      );
    },
    { unsafe: !0 },
  );
var gu = ee,
  du = pt,
  vu = C,
  yu = rr,
  Au = fr;
Xn("match", 1, function (e, t, n) {
  return [
    function (t) {
      var n = vu(this),
        r = null == t ? void 0 : t[e];
      return void 0 !== r ? r.call(t, n) : new RegExp(t)[e](String(n));
    },
    function (e) {
      var r = n(t, e, this);
      if (r.done) return r.value;
      var u = gu(e),
        i = String(this);
      if (!u.global) return Au(u, i);
      var o = u.unicode;
      u.lastIndex = 0;
      for (var a, s = [], l = 0; null !== (a = Au(u, i)); ) {
        var c = String(a[0]);
        (s[l] = c), "" === c && (u.lastIndex = yu(i, du(u.lastIndex), o)), l++;
      }
      return 0 === l ? null : s;
    },
  ];
});
var mu = g,
  ku = W.f,
  Eu = Function.prototype,
  xu = Eu.toString,
  Fu = /^\s*function ([^ (]*)/;
mu &&
  !("name" in Eu) &&
  ku(Eu, "name", {
    configurable: !0,
    get: function () {
      try {
        return xu.call(this).match(Fu)[1];
      } catch (e) {
        return "";
      }
    },
  });
var bu = D,
  Cu = function (e, t) {
    var n = [][e];
    return (
      !!n &&
      bu(function () {
        n.call(
          null,
          t ||
            function () {
              throw 1;
            },
          1,
        );
      })
    );
  },
  wu = en,
  Bu = S,
  Su = [].join,
  _u = b != Object,
  Tu = Cu("join", ",");
wu(
  { target: "Array", proto: !0, forced: _u || !Tu },
  {
    join: function (e) {
      return Su.call(Bu(this), void 0 === e ? "," : e);
    },
  },
);
var Ou = function (e) {
    if ("function" != typeof e)
      throw TypeError(String(e) + " is not a function");
    return e;
  },
  Ru = ee,
  Iu = Ou,
  ju = Pn("species"),
  zu = function (e, t) {
    var n,
      r = Ru(e).constructor;
    return void 0 === r || null == (n = Ru(r)[ju]) ? t : Iu(n);
  },
  $u = Xn,
  Pu = Tr,
  Lu = ee,
  Mu = C,
  Nu = zu,
  Uu = rr,
  qu = pt,
  Zu = fr,
  Gu = dn,
  Hu = rn.UNSUPPORTED_Y,
  Qu = [].push,
  Vu = Math.min;
$u(
  "split",
  2,
  function (e, t, n) {
    var r;
    return (
      (r =
        "c" == "abbc".split(/(b)*/)[1] ||
        4 != "test".split(/(?:)/, -1).length ||
        2 != "ab".split(/(?:ab)*/).length ||
        4 != ".".split(/(.?)(.?)/).length ||
        ".".split(/()()/).length > 1 ||
        "".split(/.?/).length
          ? function (e, n) {
              var r = String(Mu(this)),
                u = void 0 === n ? 4294967295 : n >>> 0;
              if (0 === u) return [];
              if (void 0 === e) return [r];
              if (!Pu(e)) return t.call(r, e, u);
              for (
                var i,
                  o,
                  a,
                  s = [],
                  l =
                    (e.ignoreCase ? "i" : "") +
                    (e.multiline ? "m" : "") +
                    (e.unicode ? "u" : "") +
                    (e.sticky ? "y" : ""),
                  c = 0,
                  f = new RegExp(e.source, l + "g");
                (i = Gu.call(f, r)) &&
                !(
                  (o = f.lastIndex) > c &&
                  (s.push(r.slice(c, i.index)),
                  i.length > 1 && i.index < r.length && Qu.apply(s, i.slice(1)),
                  (a = i[0].length),
                  (c = o),
                  s.length >= u)
                );

              )
                f.lastIndex === i.index && f.lastIndex++;
              return (
                c === r.length
                  ? (!a && f.test("")) || s.push("")
                  : s.push(r.slice(c)),
                s.length > u ? s.slice(0, u) : s
              );
            }
          : "0".split(void 0, 0).length
            ? function (e, n) {
                return void 0 === e && 0 === n ? [] : t.call(this, e, n);
              }
            : t),
      [
        function (t, n) {
          var u = Mu(this),
            i = null == t ? void 0 : t[e];
          return void 0 !== i ? i.call(t, u, n) : r.call(String(u), t, n);
        },
        function (e, u) {
          var i = n(r, e, this, u, r !== t);
          if (i.done) return i.value;
          var o = Lu(e),
            a = String(this),
            s = Nu(o, RegExp),
            l = o.unicode,
            c =
              (o.ignoreCase ? "i" : "") +
              (o.multiline ? "m" : "") +
              (o.unicode ? "u" : "") +
              (Hu ? "g" : "y"),
            f = new s(Hu ? "^(?:" + o.source + ")" : o, c),
            p = void 0 === u ? 4294967295 : u >>> 0;
          if (0 === p) return [];
          if (0 === a.length) return null === Zu(f, a) ? [a] : [];
          for (var h = 0, D = 0, g = []; D < a.length; ) {
            f.lastIndex = Hu ? 0 : D;
            var d,
              v = Zu(f, Hu ? a.slice(D) : a);
            if (
              null === v ||
              (d = Vu(qu(f.lastIndex + (Hu ? D : 0)), a.length)) === h
            )
              D = Uu(a, D, l);
            else {
              if ((g.push(a.slice(h, D)), g.length === p)) return g;
              for (var y = 1; y <= v.length - 1; y++)
                if ((g.push(v[y]), g.length === p)) return g;
              D = h = d;
            }
          }
          return g.push(a.slice(h)), g;
        },
      ]
    );
  },
  Hu,
);
var Yu = C,
  Ku = "[\t\n\v\f\r                　\u2028\u2029\ufeff]",
  Xu = RegExp("^" + Ku + Ku + "*"),
  Wu = RegExp(Ku + Ku + "*$"),
  Ju = function (e) {
    return function (t) {
      var n = String(Yu(t));
      return (
        1 & e && (n = n.replace(Xu, "")), 2 & e && (n = n.replace(Wu, "")), n
      );
    };
  },
  ei = { start: Ju(1), end: Ju(2), trim: Ju(3) },
  ti = D,
  ni = "\t\n\v\f\r                　\u2028\u2029\ufeff",
  ri = function (e) {
    return ti(function () {
      return !!ni[e]() || "​᠎" != "​᠎"[e]() || ni[e].name !== e;
    });
  },
  ui = ei.trim;
en(
  { target: "String", proto: !0, forced: ri("trim") },
  {
    trim: function () {
      return ui(this);
    },
  },
);
var ii = {
    CSSRuleList: 0,
    CSSStyleDeclaration: 0,
    CSSValueList: 0,
    ClientRectList: 0,
    DOMRectList: 0,
    DOMStringList: 0,
    DOMTokenList: 1,
    DataTransferItemList: 0,
    FileList: 0,
    HTMLAllCollection: 0,
    HTMLCollection: 0,
    HTMLFormElement: 0,
    HTMLSelectElement: 0,
    MediaList: 0,
    MimeTypeArray: 0,
    NamedNodeMap: 0,
    NodeList: 1,
    PaintRequestList: 0,
    Plugin: 0,
    PluginArray: 0,
    SVGLengthList: 0,
    SVGNumberList: 0,
    SVGPathSegList: 0,
    SVGPointList: 0,
    SVGStringList: 0,
    SVGTransformList: 0,
    SourceBufferList: 0,
    StyleSheetList: 0,
    TextTrackCueList: 0,
    TextTrackList: 0,
    TouchList: 0,
  },
  oi = Ou,
  ai = function (e, t, n) {
    if ((oi(e), void 0 === t)) return e;
    switch (n) {
      case 0:
        return function () {
          return e.call(t);
        };
      case 1:
        return function (n) {
          return e.call(t, n);
        };
      case 2:
        return function (n, r) {
          return e.call(t, n, r);
        };
      case 3:
        return function (n, r, u) {
          return e.call(t, n, r, u);
        };
    }
    return function () {
      return e.apply(t, arguments);
    };
  },
  si = E,
  li =
    Array.isArray ||
    function (e) {
      return "Array" == si(e);
    },
  ci = _,
  fi = li,
  pi = Pn("species"),
  hi = function (e, t) {
    var n;
    return (
      fi(e) &&
        ("function" != typeof (n = e.constructor) ||
        (n !== Array && !fi(n.prototype))
          ? ci(n) && null === (n = n[pi]) && (n = void 0)
          : (n = void 0)),
      new (void 0 === n ? Array : n)(0 === t ? 0 : t)
    );
  },
  Di = ai,
  gi = b,
  di = I,
  vi = pt,
  yi = hi,
  Ai = [].push,
  mi = function (e) {
    var t = 1 == e,
      n = 2 == e,
      r = 3 == e,
      u = 4 == e,
      i = 6 == e,
      o = 7 == e,
      a = 5 == e || i;
    return function (s, l, c, f) {
      for (
        var p,
          h,
          D = di(s),
          g = gi(D),
          d = Di(l, c, 3),
          v = vi(g.length),
          y = 0,
          A = f || yi,
          m = t ? A(s, v) : n || o ? A(s, 0) : void 0;
        v > y;
        y++
      )
        if ((a || y in g) && ((h = d((p = g[y]), y, D)), e))
          if (t) m[y] = h;
          else if (h)
            switch (e) {
              case 3:
                return !0;
              case 5:
                return p;
              case 6:
                return y;
              case 2:
                Ai.call(m, p);
            }
          else
            switch (e) {
              case 4:
                return !1;
              case 7:
                Ai.call(m, p);
            }
      return i ? -1 : r || u ? u : m;
    };
  },
  ki = {
    forEach: mi(0),
    map: mi(1),
    filter: mi(2),
    some: mi(3),
    every: mi(4),
    find: mi(5),
    findIndex: mi(6),
    filterOut: mi(7),
  },
  Ei = ki.forEach,
  xi = p,
  Fi = ii,
  bi = Cu("forEach")
    ? [].forEach
    : function (e) {
        return Ei(this, e, arguments.length > 1 ? arguments[1] : void 0);
      },
  Ci = se;
for (var wi in Fi) {
  var Bi = xi[wi],
    Si = Bi && Bi.prototype;
  if (Si && Si.forEach !== bi)
    try {
      Ci(Si, "forEach", bi);
    } catch (e) {
      Si.forEach = bi;
    }
}
var _i = {};
_i[Pn("toStringTag")] = "z";
var Ti = "[object z]" === String(_i),
  Oi = Ti,
  Ri = E,
  Ii = Pn("toStringTag"),
  ji =
    "Arguments" ==
    Ri(
      (function () {
        return arguments;
      })(),
    ),
  zi = Oi
    ? Ri
    : function (e) {
        var t, n, r;
        return void 0 === e
          ? "Undefined"
          : null === e
            ? "Null"
            : "string" ==
                typeof (n = (function (e, t) {
                  try {
                    return e[t];
                  } catch (e) {}
                })((t = Object(e)), Ii))
              ? n
              : ji
                ? Ri(t)
                : "Object" == (r = Ri(t)) && "function" == typeof t.callee
                  ? "Arguments"
                  : r;
      },
  $i = zi,
  Pi = Ti
    ? {}.toString
    : function () {
        return "[object " + $i(this) + "]";
      },
  Li = Ti,
  Mi = le.exports,
  Ni = Pi;
Li || Mi(Object.prototype, "toString", Ni, { unsafe: !0 });
var Ui = p.Promise,
  qi = le.exports,
  Zi = W.f,
  Gi = $,
  Hi = Pn("toStringTag"),
  Qi = function (e, t, n) {
    e &&
      !Gi((e = n ? e : e.prototype), Hi) &&
      Zi(e, Hi, { configurable: !0, value: t });
  },
  Vi = {},
  Yi = Vi,
  Ki = Pn("iterator"),
  Xi = Array.prototype,
  Wi = zi,
  Ji = Vi,
  eo = Pn("iterator"),
  to = ee,
  no = ee,
  ro = function (e) {
    return void 0 !== e && (Yi.Array === e || Xi[Ki] === e);
  },
  uo = pt,
  io = ai,
  oo = function (e) {
    if (null != e) return e[eo] || e["@@iterator"] || Ji[Wi(e)];
  },
  ao = function (e) {
    var t = e.return;
    if (void 0 !== t) return to(t.call(e)).value;
  },
  so = function (e, t) {
    (this.stopped = e), (this.result = t);
  },
  lo = Pn("iterator"),
  co = !1;
try {
  var fo = 0,
    po = {
      next: function () {
        return { done: !!fo++ };
      },
      return: function () {
        co = !0;
      },
    };
  (po[lo] = function () {
    return this;
  }),
    Array.from(po, function () {
      throw 2;
    });
} catch (e) {}
var ho,
  Do,
  go,
  vo = it("document", "documentElement"),
  yo = /(?:iphone|ipod|ipad).*applewebkit/i.test(An),
  Ao = "process" == E(p.process),
  mo = p,
  ko = D,
  Eo = ai,
  xo = vo,
  Fo = N,
  bo = yo,
  Co = Ao,
  wo = mo.location,
  Bo = mo.setImmediate,
  So = mo.clearImmediate,
  _o = mo.process,
  To = mo.MessageChannel,
  Oo = mo.Dispatch,
  Ro = 0,
  Io = {},
  jo = function (e) {
    if (Io.hasOwnProperty(e)) {
      var t = Io[e];
      delete Io[e], t();
    }
  },
  zo = function (e) {
    return function () {
      jo(e);
    };
  },
  $o = function (e) {
    jo(e.data);
  },
  Po = function (e) {
    mo.postMessage(e + "", wo.protocol + "//" + wo.host);
  };
(Bo && So) ||
  ((Bo = function (e) {
    for (var t = [], n = 1; arguments.length > n; ) t.push(arguments[n++]);
    return (
      (Io[++Ro] = function () {
        ("function" == typeof e ? e : Function(e)).apply(void 0, t);
      }),
      ho(Ro),
      Ro
    );
  }),
  (So = function (e) {
    delete Io[e];
  }),
  Co
    ? (ho = function (e) {
        _o.nextTick(zo(e));
      })
    : Oo && Oo.now
      ? (ho = function (e) {
          Oo.now(zo(e));
        })
      : To && !bo
        ? ((go = (Do = new To()).port2),
          (Do.port1.onmessage = $o),
          (ho = Eo(go.postMessage, go, 1)))
        : mo.addEventListener &&
            "function" == typeof postMessage &&
            !mo.importScripts &&
            wo &&
            "file:" !== wo.protocol &&
            !ko(Po)
          ? ((ho = Po), mo.addEventListener("message", $o, !1))
          : (ho =
              "onreadystatechange" in Fo("script")
                ? function (e) {
                    xo.appendChild(Fo("script")).onreadystatechange =
                      function () {
                        xo.removeChild(this), jo(e);
                      };
                  }
                : function (e) {
                    setTimeout(zo(e), 0);
                  }));
var Lo,
  Mo,
  No,
  Uo,
  qo,
  Zo,
  Go,
  Ho,
  Qo = { set: Bo, clear: So },
  Vo = /web0s(?!.*chrome)/i.test(An),
  Yo = p,
  Ko = h.f,
  Xo = Qo.set,
  Wo = yo,
  Jo = Vo,
  ea = Ao,
  ta = Yo.MutationObserver || Yo.WebKitMutationObserver,
  na = Yo.document,
  ra = Yo.process,
  ua = Yo.Promise,
  ia = Ko(Yo, "queueMicrotask"),
  oa = ia && ia.value;
oa ||
  ((Lo = function () {
    var e, t;
    for (ea && (e = ra.domain) && e.exit(); Mo; ) {
      (t = Mo.fn), (Mo = Mo.next);
      try {
        t();
      } catch (e) {
        throw (Mo ? Uo() : (No = void 0), e);
      }
    }
    (No = void 0), e && e.enter();
  }),
  Wo || ea || Jo || !ta || !na
    ? ua && ua.resolve
      ? (((Go = ua.resolve(void 0)).constructor = ua),
        (Ho = Go.then),
        (Uo = function () {
          Ho.call(Go, Lo);
        }))
      : (Uo = ea
          ? function () {
              ra.nextTick(Lo);
            }
          : function () {
              Xo.call(Yo, Lo);
            })
    : ((qo = !0),
      (Zo = na.createTextNode("")),
      new ta(Lo).observe(Zo, { characterData: !0 }),
      (Uo = function () {
        Zo.data = qo = !qo;
      })));
var aa =
    oa ||
    function (e) {
      var t = { fn: e, next: void 0 };
      No && (No.next = t), Mo || ((Mo = t), Uo()), (No = t);
    },
  sa = {},
  la = Ou,
  ca = function (e) {
    var t, n;
    (this.promise = new e(function (e, r) {
      if (void 0 !== t || void 0 !== n)
        throw TypeError("Bad Promise constructor");
      (t = e), (n = r);
    })),
      (this.resolve = la(t)),
      (this.reject = la(n));
  };
sa.f = function (e) {
  return new ca(e);
};
var fa,
  pa,
  ha,
  Da,
  ga = ee,
  da = _,
  va = sa,
  ya = p,
  Aa = "object" == typeof window,
  ma = en,
  ka = p,
  Ea = it,
  xa = Ui,
  Fa = le.exports,
  ba = function (e, t, n) {
    for (var r in t) qi(e, r, t[r], n);
    return e;
  },
  Ca = br,
  wa = Qi,
  Ba = zr,
  Sa = _,
  _a = Ou,
  Ta = function (e, t, n) {
    if (!(e instanceof t))
      throw TypeError("Incorrect " + (n ? n + " " : "") + "invocation");
    return e;
  },
  Oa = ve,
  Ra = function (e, t, n) {
    var r,
      u,
      i,
      o,
      a,
      s,
      l,
      c = n && n.that,
      f = !(!n || !n.AS_ENTRIES),
      p = !(!n || !n.IS_ITERATOR),
      h = !(!n || !n.INTERRUPTED),
      D = io(t, c, 1 + f + h),
      g = function (e) {
        return r && ao(r), new so(!0, e);
      },
      d = function (e) {
        return f
          ? (no(e), h ? D(e[0], e[1], g) : D(e[0], e[1]))
          : h
            ? D(e, g)
            : D(e);
      };
    if (p) r = e;
    else {
      if ("function" != typeof (u = oo(e)))
        throw TypeError("Target is not iterable");
      if (ro(u)) {
        for (i = 0, o = uo(e.length); o > i; i++)
          if ((a = d(e[i])) && a instanceof so) return a;
        return new so(!1);
      }
      r = u.call(e);
    }
    for (s = r.next; !(l = s.call(r)).done; ) {
      try {
        a = d(l.value);
      } catch (e) {
        throw (ao(r), e);
      }
      if ("object" == typeof a && a && a instanceof so) return a;
    }
    return new so(!1);
  },
  Ia = function (e, t) {
    if (!t && !co) return !1;
    var n = !1;
    try {
      var r = {};
      (r[lo] = function () {
        return {
          next: function () {
            return { done: (n = !0) };
          },
        };
      }),
        e(r);
    } catch (e) {}
    return n;
  },
  ja = zu,
  za = Qo.set,
  $a = aa,
  Pa = function (e, t) {
    if ((ga(e), da(t) && t.constructor === e)) return t;
    var n = va.f(e);
    return (0, n.resolve)(t), n.promise;
  },
  La = function (e, t) {
    var n = ya.console;
    n && n.error && (1 === arguments.length ? n.error(e) : n.error(e, t));
  },
  Ma = sa,
  Na = function (e) {
    try {
      return { error: !1, value: e() };
    } catch (e) {
      return { error: !0, value: e };
    }
  },
  Ua = Qe,
  qa = Ht,
  Za = Aa,
  Ga = Ao,
  Ha = Fn,
  Qa = Pn("species"),
  Va = "Promise",
  Ya = Ua.get,
  Ka = Ua.set,
  Xa = Ua.getterFor(Va),
  Wa = xa && xa.prototype,
  Ja = xa,
  es = Wa,
  ts = ka.TypeError,
  ns = ka.document,
  rs = ka.process,
  us = Ma.f,
  is = us,
  os = !!(ns && ns.createEvent && ka.dispatchEvent),
  as = "function" == typeof PromiseRejectionEvent,
  ss = !1,
  ls = qa(Va, function () {
    var e = Oa(Ja) !== String(Ja);
    if (!e && 66 === Ha) return !0;
    if (Ha >= 51 && /native code/.test(Ja)) return !1;
    var t = new Ja(function (e) {
        e(1);
      }),
      n = function (e) {
        e(
          function () {},
          function () {},
        );
      };
    return (
      ((t.constructor = {})[Qa] = n),
      !(ss = t.then(function () {}) instanceof n) || (!e && Za && !as)
    );
  }),
  cs =
    ls ||
    !Ia(function (e) {
      Ja.all(e).catch(function () {});
    }),
  fs = function (e) {
    var t;
    return !(!Sa(e) || "function" != typeof (t = e.then)) && t;
  },
  ps = function (e, t) {
    if (!e.notified) {
      e.notified = !0;
      var n = e.reactions;
      $a(function () {
        for (var r = e.value, u = 1 == e.state, i = 0; n.length > i; ) {
          var o,
            a,
            s,
            l = n[i++],
            c = u ? l.ok : l.fail,
            f = l.resolve,
            p = l.reject,
            h = l.domain;
          try {
            c
              ? (u || (2 === e.rejection && ds(e), (e.rejection = 1)),
                !0 === c
                  ? (o = r)
                  : (h && h.enter(), (o = c(r)), h && (h.exit(), (s = !0))),
                o === l.promise
                  ? p(ts("Promise-chain cycle"))
                  : (a = fs(o))
                    ? a.call(o, f, p)
                    : f(o))
              : p(r);
          } catch (e) {
            h && !s && h.exit(), p(e);
          }
        }
        (e.reactions = []), (e.notified = !1), t && !e.rejection && Ds(e);
      });
    }
  },
  hs = function (e, t, n) {
    var r, u;
    os
      ? (((r = ns.createEvent("Event")).promise = t),
        (r.reason = n),
        r.initEvent(e, !1, !0),
        ka.dispatchEvent(r))
      : (r = { promise: t, reason: n }),
      !as && (u = ka["on" + e])
        ? u(r)
        : "unhandledrejection" === e && La("Unhandled promise rejection", n);
  },
  Ds = function (e) {
    za.call(ka, function () {
      var t,
        n = e.facade,
        r = e.value;
      if (
        gs(e) &&
        ((t = Na(function () {
          Ga
            ? rs.emit("unhandledRejection", r, n)
            : hs("unhandledrejection", n, r);
        })),
        (e.rejection = Ga || gs(e) ? 2 : 1),
        t.error)
      )
        throw t.value;
    });
  },
  gs = function (e) {
    return 1 !== e.rejection && !e.parent;
  },
  ds = function (e) {
    za.call(ka, function () {
      var t = e.facade;
      Ga ? rs.emit("rejectionHandled", t) : hs("rejectionhandled", t, e.value);
    });
  },
  vs = function (e, t, n) {
    return function (r) {
      e(t, r, n);
    };
  },
  ys = function (e, t, n) {
    e.done ||
      ((e.done = !0), n && (e = n), (e.value = t), (e.state = 2), ps(e, !0));
  },
  As = function (e, t, n) {
    if (!e.done) {
      (e.done = !0), n && (e = n);
      try {
        if (e.facade === t) throw ts("Promise can't be resolved itself");
        var r = fs(t);
        r
          ? $a(function () {
              var n = { done: !1 };
              try {
                r.call(t, vs(As, n, e), vs(ys, n, e));
              } catch (t) {
                ys(n, t, e);
              }
            })
          : ((e.value = t), (e.state = 1), ps(e, !1));
      } catch (t) {
        ys({ done: !1 }, t, e);
      }
    }
  };
if (
  ls &&
  ((es = (Ja = function (e) {
    Ta(this, Ja, Va), _a(e), fa.call(this);
    var t = Ya(this);
    try {
      e(vs(As, t), vs(ys, t));
    } catch (e) {
      ys(t, e);
    }
  }).prototype),
  ((fa = function (e) {
    Ka(this, {
      type: Va,
      done: !1,
      notified: !1,
      parent: !1,
      reactions: [],
      rejection: !1,
      state: 0,
      value: void 0,
    });
  }).prototype = ba(es, {
    then: function (e, t) {
      var n = Xa(this),
        r = us(ja(this, Ja));
      return (
        (r.ok = "function" != typeof e || e),
        (r.fail = "function" == typeof t && t),
        (r.domain = Ga ? rs.domain : void 0),
        (n.parent = !0),
        n.reactions.push(r),
        0 != n.state && ps(n, !1),
        r.promise
      );
    },
    catch: function (e) {
      return this.then(void 0, e);
    },
  })),
  (pa = function () {
    var e = new fa(),
      t = Ya(e);
    (this.promise = e), (this.resolve = vs(As, t)), (this.reject = vs(ys, t));
  }),
  (Ma.f = us =
    function (e) {
      return e === Ja || e === ha ? new pa(e) : is(e);
    }),
  "function" == typeof xa && Wa !== Object.prototype)
) {
  (Da = Wa.then),
    ss ||
      (Fa(
        Wa,
        "then",
        function (e, t) {
          var n = this;
          return new Ja(function (e, t) {
            Da.call(n, e, t);
          }).then(e, t);
        },
        { unsafe: !0 },
      ),
      Fa(Wa, "catch", es.catch, { unsafe: !0 }));
  try {
    delete Wa.constructor;
  } catch (e) {}
  Ca && Ca(Wa, es);
}
ma({ global: !0, wrap: !0, forced: ls }, { Promise: Ja }),
  wa(Ja, Va, !1),
  Ba(Va),
  (ha = Ea(Va)),
  ma(
    { target: Va, stat: !0, forced: ls },
    {
      reject: function (e) {
        var t = us(this);
        return t.reject.call(void 0, e), t.promise;
      },
    },
  ),
  ma(
    { target: Va, stat: !0, forced: ls },
    {
      resolve: function (e) {
        return Pa(this, e);
      },
    },
  ),
  ma(
    { target: Va, stat: !0, forced: cs },
    {
      all: function (e) {
        var t = this,
          n = us(t),
          r = n.resolve,
          u = n.reject,
          i = Na(function () {
            var n = _a(t.resolve),
              i = [],
              o = 0,
              a = 1;
            Ra(e, function (e) {
              var s = o++,
                l = !1;
              i.push(void 0),
                a++,
                n.call(t, e).then(function (e) {
                  l || ((l = !0), (i[s] = e), --a || r(i));
                }, u);
            }),
              --a || r(i);
          });
        return i.error && u(i.value), n.promise;
      },
      race: function (e) {
        var t = this,
          n = us(t),
          r = n.reject,
          u = Na(function () {
            var u = _a(t.resolve);
            Ra(e, function (e) {
              u.call(t, e).then(n.resolve, r);
            });
          });
        return u.error && r(u.value), n.promise;
      },
    },
  );
var ms = O,
  ks = W,
  Es = m,
  xs = function (e, t, n) {
    var r = ms(t);
    r in e ? ks.f(e, r, Es(0, n)) : (e[r] = n);
  },
  Fs = D,
  bs = Fn,
  Cs = Pn("species"),
  ws = function (e) {
    return (
      bs >= 51 ||
      !Fs(function () {
        var t = [];
        return (
          ((t.constructor = {})[Cs] = function () {
            return { foo: 1 };
          }),
          1 !== t[e](Boolean).foo
        );
      })
    );
  },
  Bs = en,
  Ss = _,
  _s = li,
  Ts = dt,
  Os = pt,
  Rs = S,
  Is = xs,
  js = Pn,
  zs = ws("slice"),
  $s = js("species"),
  Ps = [].slice,
  Ls = Math.max;
Bs(
  { target: "Array", proto: !0, forced: !zs },
  {
    slice: function (e, t) {
      var n,
        r,
        u,
        i = Rs(this),
        o = Os(i.length),
        a = Ts(e, o),
        s = Ts(void 0 === t ? o : t, o);
      if (
        _s(i) &&
        ("function" != typeof (n = i.constructor) ||
        (n !== Array && !_s(n.prototype))
          ? Ss(n) && null === (n = n[$s]) && (n = void 0)
          : (n = void 0),
        n === Array || void 0 === n)
      )
        return Ps.call(i, a, s);
      for (
        r = new (void 0 === n ? Array : n)(Ls(s - a, 0)), u = 0;
        a < s;
        a++, u++
      )
        a in i && Is(r, u, i[a]);
      return (r.length = u), r;
    },
  },
);
var Ms,
  Ns = Ct,
  Us = wt,
  qs =
    Object.keys ||
    function (e) {
      return Ns(e, Us);
    },
  Zs = W,
  Gs = ee,
  Hs = qs,
  Qs = g
    ? Object.defineProperties
    : function (e, t) {
        Gs(e);
        for (var n, r = Hs(t), u = r.length, i = 0; u > i; )
          Zs.f(e, (n = r[i++]), t[n]);
        return e;
      },
  Vs = ee,
  Ys = Qs,
  Ks = wt,
  Xs = Re,
  Ws = vo,
  Js = N,
  el = Oe("IE_PROTO"),
  tl = function () {},
  nl = function (e) {
    return "<script>" + e + "</script>";
  },
  rl = function () {
    try {
      Ms = document.domain && new ActiveXObject("htmlfile");
    } catch (e) {}
    var e, t;
    rl = Ms
      ? (function (e) {
          e.write(nl("")), e.close();
          var t = e.parentWindow.Object;
          return (e = null), t;
        })(Ms)
      : (((t = Js("iframe")).style.display = "none"),
        Ws.appendChild(t),
        (t.src = String("javascript:")),
        (e = t.contentWindow.document).open(),
        e.write(nl("document.F=Object")),
        e.close(),
        e.F);
    for (var n = Ks.length; n--; ) delete rl.prototype[Ks[n]];
    return rl();
  };
Xs[el] = !0;
var ul =
    Object.create ||
    function (e, t) {
      var n;
      return (
        null !== e
          ? ((tl.prototype = Vs(e)),
            (n = new tl()),
            (tl.prototype = null),
            (n[el] = e))
          : (n = rl()),
        void 0 === t ? n : Ys(n, t)
      );
    },
  il = ul,
  ol = W,
  al = Pn("unscopables"),
  sl = Array.prototype;
null == sl[al] && ol.f(sl, al, { configurable: !0, value: il(null) });
var ll,
  cl,
  fl,
  pl = function (e) {
    sl[al][e] = !0;
  },
  hl = !D(function () {
    function e() {}
    return (
      (e.prototype.constructor = null),
      Object.getPrototypeOf(new e()) !== e.prototype
    );
  }),
  Dl = $,
  gl = I,
  dl = hl,
  vl = Oe("IE_PROTO"),
  yl = Object.prototype,
  Al = dl
    ? Object.getPrototypeOf
    : function (e) {
        return (
          (e = gl(e)),
          Dl(e, vl)
            ? e[vl]
            : "function" == typeof e.constructor && e instanceof e.constructor
              ? e.constructor.prototype
              : e instanceof Object
                ? yl
                : null
        );
      },
  ml = D,
  kl = Al,
  El = se,
  xl = $,
  Fl = Pn("iterator"),
  bl = !1;
[].keys &&
  ("next" in (fl = [].keys())
    ? (cl = kl(kl(fl))) !== Object.prototype && (ll = cl)
    : (bl = !0)),
  (null == ll ||
    ml(function () {
      var e = {};
      return ll[Fl].call(e) !== e;
    })) &&
    (ll = {}),
  xl(ll, Fl) ||
    El(ll, Fl, function () {
      return this;
    });
var Cl = { IteratorPrototype: ll, BUGGY_SAFARI_ITERATORS: bl },
  wl = Cl.IteratorPrototype,
  Bl = ul,
  Sl = m,
  _l = Qi,
  Tl = Vi,
  Ol = function () {
    return this;
  },
  Rl = en,
  Il = function (e, t, n) {
    var r = t + " Iterator";
    return (
      (e.prototype = Bl(wl, { next: Sl(1, n) })), _l(e, r, !1), (Tl[r] = Ol), e
    );
  },
  jl = Al,
  zl = br,
  $l = Qi,
  Pl = se,
  Ll = le.exports,
  Ml = Vi,
  Nl = Cl.IteratorPrototype,
  Ul = Cl.BUGGY_SAFARI_ITERATORS,
  ql = Pn("iterator"),
  Zl = function () {
    return this;
  },
  Gl = function (e, t, n, r, u, i, o) {
    Il(n, t, r);
    var a,
      s,
      l,
      c = function (e) {
        if (e === u && g) return g;
        if (!Ul && e in h) return h[e];
        switch (e) {
          case "keys":
          case "values":
          case "entries":
            return function () {
              return new n(this, e);
            };
        }
        return function () {
          return new n(this);
        };
      },
      f = t + " Iterator",
      p = !1,
      h = e.prototype,
      D = h[ql] || h["@@iterator"] || (u && h[u]),
      g = (!Ul && D) || c(u),
      d = ("Array" == t && h.entries) || D;
    if (
      (d &&
        ((a = jl(d.call(new e()))),
        Nl !== Object.prototype &&
          a.next &&
          (jl(a) !== Nl &&
            (zl ? zl(a, Nl) : "function" != typeof a[ql] && Pl(a, ql, Zl)),
          $l(a, f, !0))),
      "values" == u &&
        D &&
        "values" !== D.name &&
        ((p = !0),
        (g = function () {
          return D.call(this);
        })),
      h[ql] !== g && Pl(h, ql, g),
      (Ml[t] = g),
      u)
    )
      if (
        ((s = {
          values: c("values"),
          keys: i ? g : c("keys"),
          entries: c("entries"),
        }),
        o)
      )
        for (l in s) (Ul || p || !(l in h)) && Ll(h, l, s[l]);
      else Rl({ target: t, proto: !0, forced: Ul || p }, s);
    return s;
  },
  Hl = S,
  Ql = pl,
  Vl = Vi,
  Yl = Qe,
  Kl = Gl,
  Xl = Yl.set,
  Wl = Yl.getterFor("Array Iterator"),
  Jl = Kl(
    Array,
    "Array",
    function (e, t) {
      Xl(this, { type: "Array Iterator", target: Hl(e), index: 0, kind: t });
    },
    function () {
      var e = Wl(this),
        t = e.target,
        n = e.kind,
        r = e.index++;
      return !t || r >= t.length
        ? ((e.target = void 0), { value: void 0, done: !0 })
        : "keys" == n
          ? { value: r, done: !1 }
          : "values" == n
            ? { value: t[r], done: !1 }
            : { value: [r, t[r]], done: !1 };
    },
    "values",
  );
(Vl.Arguments = Vl.Array), Ql("keys"), Ql("values"), Ql("entries");
var ec = tr.charAt,
  tc = Qe,
  nc = Gl,
  rc = tc.set,
  uc = tc.getterFor("String Iterator");
nc(
  String,
  "String",
  function (e) {
    rc(this, { type: "String Iterator", string: String(e), index: 0 });
  },
  function () {
    var e,
      t = uc(this),
      n = t.string,
      r = t.index;
    return r >= n.length
      ? { value: void 0, done: !0 }
      : ((e = ec(n, r)), (t.index += e.length), { value: e, done: !1 });
  },
);
var ic = p,
  oc = ii,
  ac = Jl,
  sc = se,
  lc = Pn,
  cc = lc("iterator"),
  fc = lc("toStringTag"),
  pc = ac.values;
for (var hc in oc) {
  var Dc = ic[hc],
    gc = Dc && Dc.prototype;
  if (gc) {
    if (gc[cc] !== pc)
      try {
        sc(gc, cc, pc);
      } catch (e) {
        gc[cc] = pc;
      }
    if ((gc[fc] || sc(gc, fc, hc), oc[hc]))
      for (var dc in ac)
        if (gc[dc] !== ac[dc])
          try {
            sc(gc, dc, ac[dc]);
          } catch (e) {
            gc[dc] = ac[dc];
          }
  }
}
var vc = en,
  yc = D,
  Ac = li,
  mc = _,
  kc = I,
  Ec = pt,
  xc = xs,
  Fc = hi,
  bc = ws,
  Cc = Fn,
  wc = Pn("isConcatSpreadable"),
  Bc =
    Cc >= 51 ||
    !yc(function () {
      var e = [];
      return (e[wc] = !1), e.concat()[0] !== e;
    }),
  Sc = bc("concat"),
  _c = function (e) {
    if (!mc(e)) return !1;
    var t = e[wc];
    return void 0 !== t ? !!t : Ac(e);
  };
vc(
  { target: "Array", proto: !0, forced: !Bc || !Sc },
  {
    concat: function (e) {
      var t,
        n,
        r,
        u,
        i,
        o = kc(this),
        a = Fc(o, 0),
        s = 0;
      for (t = -1, r = arguments.length; t < r; t++)
        if (_c((i = -1 === t ? o : arguments[t]))) {
          if (s + (u = Ec(i.length)) > 9007199254740991)
            throw TypeError("Maximum allowed index exceeded");
          for (n = 0; n < u; n++, s++) n in i && xc(a, s, i[n]);
        } else {
          if (s >= 9007199254740991)
            throw TypeError("Maximum allowed index exceeded");
          xc(a, s++, i);
        }
      return (a.length = s), a;
    },
  },
);
var Tc = en,
  Oc = dt,
  Rc = lt,
  Ic = pt,
  jc = I,
  zc = hi,
  $c = xs,
  Pc = ws("splice"),
  Lc = Math.max,
  Mc = Math.min;
Tc(
  { target: "Array", proto: !0, forced: !Pc },
  {
    splice: function (e, t) {
      var n,
        r,
        u,
        i,
        o,
        a,
        s = jc(this),
        l = Ic(s.length),
        c = Oc(e, l),
        f = arguments.length;
      if (
        (0 === f
          ? (n = r = 0)
          : 1 === f
            ? ((n = 0), (r = l - c))
            : ((n = f - 2), (r = Mc(Lc(Rc(t), 0), l - c))),
        l + n - r > 9007199254740991)
      )
        throw TypeError("Maximum allowed length exceeded");
      for (u = zc(s, r), i = 0; i < r; i++) (o = c + i) in s && $c(u, i, s[o]);
      if (((u.length = r), n < r)) {
        for (i = c; i < l - r; i++)
          (a = i + n), (o = i + r) in s ? (s[a] = s[o]) : delete s[a];
        for (i = l; i > l - r + n; i--) delete s[i - 1];
      } else if (n > r)
        for (i = l - r; i > c; i--)
          (a = i + n - 1), (o = i + r - 1) in s ? (s[a] = s[o]) : delete s[a];
      for (i = 0; i < n; i++) s[i + c] = arguments[i + 2];
      return (s.length = l - r + n), u;
    },
  },
);
var Nc = ki.map;
en(
  { target: "Array", proto: !0, forced: !ws("map") },
  {
    map: function (e) {
      return Nc(this, e, arguments.length > 1 ? arguments[1] : void 0);
    },
  },
);
var Uc = en,
  qc = ei.start,
  Zc = ri("trimStart"),
  Gc = Zc
    ? function () {
        return qc(this);
      }
    : "".trimStart;
Uc(
  { target: "String", proto: !0, forced: Zc },
  { trimStart: Gc, trimLeft: Gc },
);
var Hc =
    Object.is ||
    function (e, t) {
      return e === t ? 0 !== e || 1 / e == 1 / t : e != e && t != t;
    },
  Qc = ee,
  Vc = C,
  Yc = Hc,
  Kc = fr;
Xn("search", 1, function (e, t, n) {
  return [
    function (t) {
      var n = Vc(this),
        r = null == t ? void 0 : t[e];
      return void 0 !== r ? r.call(t, n) : new RegExp(t)[e](String(n));
    },
    function (e) {
      var r = n(t, e, this);
      if (r.done) return r.value;
      var u = Qc(e),
        i = String(this),
        o = u.lastIndex;
      Yc(o, 0) || (u.lastIndex = 0);
      var a = Kc(u, i);
      return Yc(u.lastIndex, o) || (u.lastIndex = o), null === a ? -1 : a.index;
    },
  ];
});
var Xc = en,
  Wc = ei.end,
  Jc = ri("trimEnd"),
  ef = Jc
    ? function () {
        return Wc(this);
      }
    : "".trimEnd;
Xc({ target: "String", proto: !0, forced: Jc }, { trimEnd: ef, trimRight: ef });
var tf = ki.filter;
en(
  { target: "Array", proto: !0, forced: !ws("filter") },
  {
    filter: function (e) {
      return tf(this, e, arguments.length > 1 ? arguments[1] : void 0);
    },
  },
);
var nf = C,
  rf = /"/g,
  uf = D,
  of = function (e, t, n, r) {
    var u = String(nf(e)),
      i = "<" + t;
    return (
      "" !== n && (i += " " + n + '="' + String(r).replace(rf, "&quot;") + '"'),
      i + ">" + u + "</" + t + ">"
    );
  };
en(
  {
    target: "String",
    proto: !0,
    forced: (function (e) {
      return uf(function () {
        var t = ""[e]('"');
        return t !== t.toLowerCase() || t.split('"').length > 3;
      });
    })("link"),
  },
  {
    link: function (e) {
      return of(this, "a", "href", e);
    },
  },
);
var af = I,
  sf = qs;
en(
  {
    target: "Object",
    stat: !0,
    forced: D(function () {
      sf(1);
    }),
  },
  {
    keys: function (e) {
      return sf(af(e));
    },
  },
);
var lf = kt.includes,
  cf = pl;
en(
  { target: "Array", proto: !0 },
  {
    includes: function (e) {
      return lf(this, e, arguments.length > 1 ? arguments[1] : void 0);
    },
  },
),
  cf("includes");
var ff = Tr,
  pf = Pn("match"),
  hf = function (e) {
    if (ff(e)) throw TypeError("The method doesn't accept regular expressions");
    return e;
  },
  Df = C;
function gf() {
  return {
    baseUrl: null,
    breaks: !1,
    extensions: null,
    gfm: !0,
    headerIds: !0,
    headerPrefix: "",
    highlight: null,
    langPrefix: "language-",
    mangle: !0,
    pedantic: !1,
    renderer: null,
    sanitize: !1,
    sanitizer: null,
    silent: !1,
    smartLists: !1,
    smartypants: !1,
    tokenizer: null,
    walkTokens: null,
    xhtml: !1,
  };
}
en(
  {
    target: "String",
    proto: !0,
    forced: !(function (e) {
      var t = /./;
      try {
        "/./"[e](t);
      } catch (n) {
        try {
          return (t[pf] = !1), "/./"[e](t);
        } catch (e) {}
      }
      return !1;
    })("includes"),
  },
  {
    includes: function (e) {
      return !!~String(Df(this)).indexOf(
        hf(e),
        arguments.length > 1 ? arguments[1] : void 0,
      );
    },
  },
);
var df = {
  baseUrl: null,
  breaks: !1,
  extensions: null,
  gfm: !0,
  headerIds: !0,
  headerPrefix: "",
  highlight: null,
  langPrefix: "language-",
  mangle: !0,
  pedantic: !1,
  renderer: null,
  sanitize: !1,
  sanitizer: null,
  silent: !1,
  smartLists: !1,
  smartypants: !1,
  tokenizer: null,
  walkTokens: null,
  xhtml: !1,
};
var vf = /[&<>"']/,
  yf = /[&<>"']/g,
  Af = /[<>"']|&(?!#?\w+;)/,
  mf = /[<>"']|&(?!#?\w+;)/g,
  kf = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" },
  Ef = function (e) {
    return kf[e];
  };
function xf(e, t) {
  if (t) {
    if (vf.test(e)) return e.replace(yf, Ef);
  } else if (Af.test(e)) return e.replace(mf, Ef);
  return e;
}
var Ff = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi;
function bf(e) {
  return e.replace(Ff, function (e, t) {
    return "colon" === (t = t.toLowerCase())
      ? ":"
      : "#" === t.charAt(0)
        ? "x" === t.charAt(1)
          ? String.fromCharCode(parseInt(t.substring(2), 16))
          : String.fromCharCode(+t.substring(1))
        : "";
  });
}
var Cf = /(^|[^\[])\^/g;
function wf(e, t) {
  (e = e.source || e), (t = t || "");
  var n = {
    replace: function (t, r) {
      return (
        (r = (r = r.source || r).replace(Cf, "$1")), (e = e.replace(t, r)), n
      );
    },
    getRegex: function () {
      return new RegExp(e, t);
    },
  };
  return n;
}
var Bf = /[^\w:]/g,
  Sf = /^$|^[a-z][a-z0-9+.-]*:|^[?#]/i;
function _f(e, t, n) {
  if (e) {
    var r;
    try {
      r = decodeURIComponent(bf(n)).replace(Bf, "").toLowerCase();
    } catch (e) {
      return null;
    }
    if (
      0 === r.indexOf("javascript:") ||
      0 === r.indexOf("vbscript:") ||
      0 === r.indexOf("data:")
    )
      return null;
  }
  t &&
    !Sf.test(n) &&
    (n = (function (e, t) {
      Tf[" " + e] ||
        (Of.test(e) ? (Tf[" " + e] = e + "/") : (Tf[" " + e] = Pf(e, "/", !0)));
      var n = -1 === (e = Tf[" " + e]).indexOf(":");
      return "//" === t.substring(0, 2)
        ? n
          ? t
          : e.replace(Rf, "$1") + t
        : "/" === t.charAt(0)
          ? n
            ? t
            : e.replace(If, "$1") + t
          : e + t;
    })(t, n));
  try {
    n = encodeURI(n).replace(/%25/g, "%");
  } catch (e) {
    return null;
  }
  return n;
}
var Tf = {},
  Of = /^[^:]+:\/*[^/]*$/,
  Rf = /^([^:]+:)[\s\S]*$/,
  If = /^([^:]+:\/*[^/]*)[\s\S]*$/;
var jf = { exec: function () {} };
function zf(e) {
  for (var t, n, r = 1; r < arguments.length; r++)
    for (n in (t = arguments[r]))
      Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n]);
  return e;
}
function $f(e, t) {
  var n = e
      .replace(/\|/g, function (e, t, n) {
        for (var r = !1, u = t; --u >= 0 && "\\" === n[u]; ) r = !r;
        return r ? "|" : " |";
      })
      .split(/ \|/),
    r = 0;
  if (
    (n[0].trim() || n.shift(),
    n.length > 0 && !n[n.length - 1].trim() && n.pop(),
    n.length > t)
  )
    n.splice(t);
  else for (; n.length < t; ) n.push("");
  for (; r < n.length; r++) n[r] = n[r].trim().replace(/\\\|/g, "|");
  return n;
}
function Pf(e, t, n) {
  var r = e.length;
  if (0 === r) return "";
  for (var u = 0; u < r; ) {
    var i = e.charAt(r - u - 1);
    if (i !== t || n) {
      if (i === t || !n) break;
      u++;
    } else u++;
  }
  return e.substr(0, r - u);
}
function Lf(e) {
  e &&
    e.sanitize &&
    !e.silent &&
    console.warn(
      "marked(): sanitize and sanitizer parameters are deprecated since version 0.7.0, should not be used and will be removed in the future. Read more here: https://marked.js.org/#/USING_ADVANCED.md#options",
    );
}
function Mf(e, t) {
  if (t < 1) return "";
  for (var n = ""; t > 1; ) 1 & t && (n += e), (t >>= 1), (e += e);
  return n + e;
}
function Nf(e, t, n, r) {
  var u = t.href,
    i = t.title ? xf(t.title) : null,
    o = e[1].replace(/\\([\[\]])/g, "$1");
  if ("!" !== e[0].charAt(0)) {
    r.state.inLink = !0;
    var a = {
      type: "link",
      raw: n,
      href: u,
      title: i,
      text: o,
      tokens: r.inlineTokens(o, []),
    };
    return (r.state.inLink = !1), a;
  }
  return { type: "image", raw: n, href: u, title: i, text: xf(o) };
}
var Uf = (function () {
    function e(n) {
      t(this, e), (this.options = n || df);
    }
    return (
      r(e, [
        {
          key: "space",
          value: function (e) {
            var t = this.rules.block.newline.exec(e);
            if (t && t[0].length > 0) return { type: "space", raw: t[0] };
          },
        },
        {
          key: "code",
          value: function (e) {
            var t = this.rules.block.code.exec(e);
            if (t) {
              var n = t[0].replace(/^ {1,4}/gm, "");
              return {
                type: "code",
                raw: t[0],
                codeBlockStyle: "indented",
                text: this.options.pedantic ? n : Pf(n, "\n"),
              };
            }
          },
        },
        {
          key: "fences",
          value: function (e) {
            var t = this.rules.block.fences.exec(e);
            if (t) {
              var n = t[0],
                r = (function (e, t) {
                  var n = e.match(/^(\s+)(?:```)/);
                  if (null === n) return t;
                  var r = n[1];
                  return t
                    .split("\n")
                    .map(function (e) {
                      var t = e.match(/^\s+/);
                      return null === t
                        ? e
                        : o(t, 1)[0].length >= r.length
                          ? e.slice(r.length)
                          : e;
                    })
                    .join("\n");
                })(n, t[3] || "");
              return {
                type: "code",
                raw: n,
                lang: t[2] ? t[2].trim() : t[2],
                text: r,
              };
            }
          },
        },
        {
          key: "heading",
          value: function (e) {
            var t = this.rules.block.heading.exec(e);
            if (t) {
              var n = t[2].trim();
              if (/#$/.test(n)) {
                var r = Pf(n, "#");
                this.options.pedantic
                  ? (n = r.trim())
                  : (r && !/ $/.test(r)) || (n = r.trim());
              }
              var u = {
                type: "heading",
                raw: t[0],
                depth: t[1].length,
                text: n,
                tokens: [],
              };
              return this.lexer.inline(u.text, u.tokens), u;
            }
          },
        },
        {
          key: "hr",
          value: function (e) {
            var t = this.rules.block.hr.exec(e);
            if (t) return { type: "hr", raw: t[0] };
          },
        },
        {
          key: "blockquote",
          value: function (e) {
            var t = this.rules.block.blockquote.exec(e);
            if (t) {
              var n = t[0].replace(/^ *> ?/gm, "");
              return {
                type: "blockquote",
                raw: t[0],
                tokens: this.lexer.blockTokens(n, []),
                text: n,
              };
            }
          },
        },
        {
          key: "list",
          value: function (e) {
            var t = this.rules.block.list.exec(e);
            if (t) {
              var n,
                r,
                u,
                i,
                o,
                a,
                s,
                c,
                f,
                p,
                h,
                D,
                g = t[1].trim(),
                d = g.length > 1,
                v = {
                  type: "list",
                  raw: "",
                  ordered: d,
                  start: d ? +g.slice(0, -1) : "",
                  loose: !1,
                  items: [],
                };
              (g = d ? "\\d{1,9}\\".concat(g.slice(-1)) : "\\".concat(g)),
                this.options.pedantic && (g = d ? g : "[*+-]");
              for (
                var y = new RegExp(
                  "^( {0,3}".concat(g, ")((?: [^\\n]*)?(?:\\n|$))"),
                );
                e &&
                ((D = !1), (t = y.exec(e))) &&
                !this.rules.block.hr.test(e);

              ) {
                if (
                  ((n = t[0]),
                  (e = e.substring(n.length)),
                  (c = t[2].split("\n", 1)[0]),
                  (f = e.split("\n", 1)[0]),
                  this.options.pedantic
                    ? ((i = 2), (h = c.trimLeft()))
                    : ((i = (i = t[2].search(/[^ ]/)) > 4 ? 1 : i),
                      (h = c.slice(i)),
                      (i += t[1].length)),
                  (a = !1),
                  !c &&
                    /^ *$/.test(f) &&
                    ((n += f + "\n"),
                    (e = e.substring(f.length + 1)),
                    (D = !0)),
                  !D)
                )
                  for (
                    var A = new RegExp(
                      "^ {0,".concat(
                        Math.min(3, i - 1),
                        "}(?:[*+-]|\\d{1,9}[.)])",
                      ),
                    );
                    e &&
                    ((c = p = e.split("\n", 1)[0]),
                    this.options.pedantic &&
                      (c = c.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ")),
                    !A.test(c));

                  ) {
                    if (c.search(/[^ ]/) >= i || !c.trim())
                      h += "\n" + c.slice(i);
                    else {
                      if (a) break;
                      h += "\n" + c;
                    }
                    a || c.trim() || (a = !0),
                      (n += p + "\n"),
                      (e = e.substring(p.length + 1));
                  }
                v.loose ||
                  (s ? (v.loose = !0) : /\n *\n *$/.test(n) && (s = !0)),
                  this.options.gfm &&
                    (r = /^\[[ xX]\] /.exec(h)) &&
                    ((u = "[ ] " !== r[0]),
                    (h = h.replace(/^\[[ xX]\] +/, ""))),
                  v.items.push({
                    type: "list_item",
                    raw: n,
                    task: !!r,
                    checked: u,
                    loose: !1,
                    text: h,
                  }),
                  (v.raw += n);
              }
              (v.items[v.items.length - 1].raw = n.trimRight()),
                (v.items[v.items.length - 1].text = h.trimRight()),
                (v.raw = v.raw.trimRight());
              var m = v.items.length;
              for (o = 0; o < m; o++) {
                (this.lexer.state.top = !1),
                  (v.items[o].tokens = this.lexer.blockTokens(
                    v.items[o].text,
                    [],
                  ));
                var k = v.items[o].tokens.filter(function (e) {
                    return "space" === e.type;
                  }),
                  E = k.every(function (e) {
                    var t,
                      n = 0,
                      r = l(e.raw.split(""));
                    try {
                      for (r.s(); !(t = r.n()).done; ) {
                        if (("\n" === t.value && (n += 1), n > 1)) return !0;
                      }
                    } catch (e) {
                      r.e(e);
                    } finally {
                      r.f();
                    }
                    return !1;
                  });
                !v.loose &&
                  k.length &&
                  E &&
                  ((v.loose = !0), (v.items[o].loose = !0));
              }
              return v;
            }
          },
        },
        {
          key: "html",
          value: function (e) {
            var t = this.rules.block.html.exec(e);
            if (t) {
              var n = {
                type: "html",
                raw: t[0],
                pre:
                  !this.options.sanitizer &&
                  ("pre" === t[1] || "script" === t[1] || "style" === t[1]),
                text: t[0],
              };
              return (
                this.options.sanitize &&
                  ((n.type = "paragraph"),
                  (n.text = this.options.sanitizer
                    ? this.options.sanitizer(t[0])
                    : xf(t[0])),
                  (n.tokens = []),
                  this.lexer.inline(n.text, n.tokens)),
                n
              );
            }
          },
        },
        {
          key: "def",
          value: function (e) {
            var t = this.rules.block.def.exec(e);
            if (t)
              return (
                t[3] && (t[3] = t[3].substring(1, t[3].length - 1)),
                {
                  type: "def",
                  tag: t[1].toLowerCase().replace(/\s+/g, " "),
                  raw: t[0],
                  href: t[2],
                  title: t[3],
                }
              );
          },
        },
        {
          key: "table",
          value: function (e) {
            var t = this.rules.block.table.exec(e);
            if (t) {
              var n = {
                type: "table",
                header: $f(t[1]).map(function (e) {
                  return { text: e };
                }),
                align: t[2].replace(/^ *|\| *$/g, "").split(/ *\| */),
                rows:
                  t[3] && t[3].trim()
                    ? t[3].replace(/\n[ \t]*$/, "").split("\n")
                    : [],
              };
              if (n.header.length === n.align.length) {
                n.raw = t[0];
                var r,
                  u,
                  i,
                  o,
                  a = n.align.length;
                for (r = 0; r < a; r++)
                  /^ *-+: *$/.test(n.align[r])
                    ? (n.align[r] = "right")
                    : /^ *:-+: *$/.test(n.align[r])
                      ? (n.align[r] = "center")
                      : /^ *:-+ *$/.test(n.align[r])
                        ? (n.align[r] = "left")
                        : (n.align[r] = null);
                for (a = n.rows.length, r = 0; r < a; r++)
                  n.rows[r] = $f(n.rows[r], n.header.length).map(function (e) {
                    return { text: e };
                  });
                for (a = n.header.length, u = 0; u < a; u++)
                  (n.header[u].tokens = []),
                    this.lexer.inlineTokens(
                      n.header[u].text,
                      n.header[u].tokens,
                    );
                for (a = n.rows.length, u = 0; u < a; u++)
                  for (o = n.rows[u], i = 0; i < o.length; i++)
                    (o[i].tokens = []),
                      this.lexer.inlineTokens(o[i].text, o[i].tokens);
                return n;
              }
            }
          },
        },
        {
          key: "lheading",
          value: function (e) {
            var t = this.rules.block.lheading.exec(e);
            if (t) {
              var n = {
                type: "heading",
                raw: t[0],
                depth: "=" === t[2].charAt(0) ? 1 : 2,
                text: t[1],
                tokens: [],
              };
              return this.lexer.inline(n.text, n.tokens), n;
            }
          },
        },
        {
          key: "paragraph",
          value: function (e) {
            var t = this.rules.block.paragraph.exec(e);
            if (t) {
              var n = {
                type: "paragraph",
                raw: t[0],
                text:
                  "\n" === t[1].charAt(t[1].length - 1)
                    ? t[1].slice(0, -1)
                    : t[1],
                tokens: [],
              };
              return this.lexer.inline(n.text, n.tokens), n;
            }
          },
        },
        {
          key: "text",
          value: function (e) {
            var t = this.rules.block.text.exec(e);
            if (t) {
              var n = { type: "text", raw: t[0], text: t[0], tokens: [] };
              return this.lexer.inline(n.text, n.tokens), n;
            }
          },
        },
        {
          key: "escape",
          value: function (e) {
            var t = this.rules.inline.escape.exec(e);
            if (t) return { type: "escape", raw: t[0], text: xf(t[1]) };
          },
        },
        {
          key: "tag",
          value: function (e) {
            var t = this.rules.inline.tag.exec(e);
            if (t)
              return (
                !this.lexer.state.inLink && /^<a /i.test(t[0])
                  ? (this.lexer.state.inLink = !0)
                  : this.lexer.state.inLink &&
                    /^<\/a>/i.test(t[0]) &&
                    (this.lexer.state.inLink = !1),
                !this.lexer.state.inRawBlock &&
                /^<(pre|code|kbd|script)(\s|>)/i.test(t[0])
                  ? (this.lexer.state.inRawBlock = !0)
                  : this.lexer.state.inRawBlock &&
                    /^<\/(pre|code|kbd|script)(\s|>)/i.test(t[0]) &&
                    (this.lexer.state.inRawBlock = !1),
                {
                  type: this.options.sanitize ? "text" : "html",
                  raw: t[0],
                  inLink: this.lexer.state.inLink,
                  inRawBlock: this.lexer.state.inRawBlock,
                  text: this.options.sanitize
                    ? this.options.sanitizer
                      ? this.options.sanitizer(t[0])
                      : xf(t[0])
                    : t[0],
                }
              );
          },
        },
        {
          key: "link",
          value: function (e) {
            var t = this.rules.inline.link.exec(e);
            if (t) {
              var n = t[2].trim();
              if (!this.options.pedantic && /^</.test(n)) {
                if (!/>$/.test(n)) return;
                var r = Pf(n.slice(0, -1), "\\");
                if ((n.length - r.length) % 2 == 0) return;
              } else {
                var u = (function (e, t) {
                  if (-1 === e.indexOf(t[1])) return -1;
                  for (var n = e.length, r = 0, u = 0; u < n; u++)
                    if ("\\" === e[u]) u++;
                    else if (e[u] === t[0]) r++;
                    else if (e[u] === t[1] && --r < 0) return u;
                  return -1;
                })(t[2], "()");
                if (u > -1) {
                  var i = (0 === t[0].indexOf("!") ? 5 : 4) + t[1].length + u;
                  (t[2] = t[2].substring(0, u)),
                    (t[0] = t[0].substring(0, i).trim()),
                    (t[3] = "");
                }
              }
              var o = t[2],
                a = "";
              if (this.options.pedantic) {
                var s = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(o);
                s && ((o = s[1]), (a = s[3]));
              } else a = t[3] ? t[3].slice(1, -1) : "";
              return (
                (o = o.trim()),
                /^</.test(o) &&
                  (o =
                    this.options.pedantic && !/>$/.test(n)
                      ? o.slice(1)
                      : o.slice(1, -1)),
                Nf(
                  t,
                  {
                    href: o ? o.replace(this.rules.inline._escapes, "$1") : o,
                    title: a ? a.replace(this.rules.inline._escapes, "$1") : a,
                  },
                  t[0],
                  this.lexer,
                )
              );
            }
          },
        },
        {
          key: "reflink",
          value: function (e, t) {
            var n;
            if (
              (n = this.rules.inline.reflink.exec(e)) ||
              (n = this.rules.inline.nolink.exec(e))
            ) {
              var r = (n[2] || n[1]).replace(/\s+/g, " ");
              if (!(r = t[r.toLowerCase()]) || !r.href) {
                var u = n[0].charAt(0);
                return { type: "text", raw: u, text: u };
              }
              return Nf(n, r, n[0], this.lexer);
            }
          },
        },
        {
          key: "emStrong",
          value: function (e, t) {
            var n =
                arguments.length > 2 && void 0 !== arguments[2]
                  ? arguments[2]
                  : "",
              r = this.rules.inline.emStrong.lDelim.exec(e);
            if (
              r &&
              (!r[3] ||
                !n.match(
                  /(?:[0-9A-Za-z\xAA\xB2\xB3\xB5\xB9\xBA\xBC-\xBE\xC0-\xD6\xD8-\xF6\xF8-\u02C1\u02C6-\u02D1\u02E0-\u02E4\u02EC\u02EE\u0370-\u0374\u0376\u0377\u037A-\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u0559\u0560-\u0588\u05D0-\u05EA\u05EF-\u05F2\u0620-\u064A\u0660-\u0669\u066E\u066F\u0671-\u06D3\u06D5\u06E5\u06E6\u06EE-\u06FC\u06FF\u0710\u0712-\u072F\u074D-\u07A5\u07B1\u07C0-\u07EA\u07F4\u07F5\u07FA\u0800-\u0815\u081A\u0824\u0828\u0840-\u0858\u0860-\u086A\u08A0-\u08B4\u08B6-\u08C7\u0904-\u0939\u093D\u0950\u0958-\u0961\u0966-\u096F\u0971-\u0980\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09BD\u09CE\u09DC\u09DD\u09DF-\u09E1\u09E6-\u09F1\u09F4-\u09F9\u09FC\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A59-\u0A5C\u0A5E\u0A66-\u0A6F\u0A72-\u0A74\u0A85-\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABD\u0AD0\u0AE0\u0AE1\u0AE6-\u0AEF\u0AF9\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B35-\u0B39\u0B3D\u0B5C\u0B5D\u0B5F-\u0B61\u0B66-\u0B6F\u0B71-\u0B77\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BD0\u0BE6-\u0BF2\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C39\u0C3D\u0C58-\u0C5A\u0C60\u0C61\u0C66-\u0C6F\u0C78-\u0C7E\u0C80\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBD\u0CDE\u0CE0\u0CE1\u0CE6-\u0CEF\u0CF1\u0CF2\u0D04-\u0D0C\u0D0E-\u0D10\u0D12-\u0D3A\u0D3D\u0D4E\u0D54-\u0D56\u0D58-\u0D61\u0D66-\u0D78\u0D7A-\u0D7F\u0D85-\u0D96\u0D9A-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0DE6-\u0DEF\u0E01-\u0E30\u0E32\u0E33\u0E40-\u0E46\u0E50-\u0E59\u0E81\u0E82\u0E84\u0E86-\u0E8A\u0E8C-\u0EA3\u0EA5\u0EA7-\u0EB0\u0EB2\u0EB3\u0EBD\u0EC0-\u0EC4\u0EC6\u0ED0-\u0ED9\u0EDC-\u0EDF\u0F00\u0F20-\u0F33\u0F40-\u0F47\u0F49-\u0F6C\u0F88-\u0F8C\u1000-\u102A\u103F-\u1049\u1050-\u1055\u105A-\u105D\u1061\u1065\u1066\u106E-\u1070\u1075-\u1081\u108E\u1090-\u1099\u10A0-\u10C5\u10C7\u10CD\u10D0-\u10FA\u10FC-\u1248\u124A-\u124D\u1250-\u1256\u1258\u125A-\u125D\u1260-\u1288\u128A-\u128D\u1290-\u12B0\u12B2-\u12B5\u12B8-\u12BE\u12C0\u12C2-\u12C5\u12C8-\u12D6\u12D8-\u1310\u1312-\u1315\u1318-\u135A\u1369-\u137C\u1380-\u138F\u13A0-\u13F5\u13F8-\u13FD\u1401-\u166C\u166F-\u167F\u1681-\u169A\u16A0-\u16EA\u16EE-\u16F8\u1700-\u170C\u170E-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176C\u176E-\u1770\u1780-\u17B3\u17D7\u17DC\u17E0-\u17E9\u17F0-\u17F9\u1810-\u1819\u1820-\u1878\u1880-\u1884\u1887-\u18A8\u18AA\u18B0-\u18F5\u1900-\u191E\u1946-\u196D\u1970-\u1974\u1980-\u19AB\u19B0-\u19C9\u19D0-\u19DA\u1A00-\u1A16\u1A20-\u1A54\u1A80-\u1A89\u1A90-\u1A99\u1AA7\u1B05-\u1B33\u1B45-\u1B4B\u1B50-\u1B59\u1B83-\u1BA0\u1BAE-\u1BE5\u1C00-\u1C23\u1C40-\u1C49\u1C4D-\u1C7D\u1C80-\u1C88\u1C90-\u1CBA\u1CBD-\u1CBF\u1CE9-\u1CEC\u1CEE-\u1CF3\u1CF5\u1CF6\u1CFA\u1D00-\u1DBF\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2070\u2071\u2074-\u2079\u207F-\u2089\u2090-\u209C\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2139\u213C-\u213F\u2145-\u2149\u214E\u2150-\u2189\u2460-\u249B\u24EA-\u24FF\u2776-\u2793\u2C00-\u2C2E\u2C30-\u2C5E\u2C60-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\u2CFD\u2D00-\u2D25\u2D27\u2D2D\u2D30-\u2D67\u2D6F\u2D80-\u2D96\u2DA0-\u2DA6\u2DA8-\u2DAE\u2DB0-\u2DB6\u2DB8-\u2DBE\u2DC0-\u2DC6\u2DC8-\u2DCE\u2DD0-\u2DD6\u2DD8-\u2DDE\u2E2F\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303C\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF\u3105-\u312F\u3131-\u318E\u3192-\u3195\u31A0-\u31BF\u31F0-\u31FF\u3220-\u3229\u3248-\u324F\u3251-\u325F\u3280-\u3289\u32B1-\u32BF\u3400-\u4DBF\u4E00-\u9FFC\uA000-\uA48C\uA4D0-\uA4FD\uA500-\uA60C\uA610-\uA62B\uA640-\uA66E\uA67F-\uA69D\uA6A0-\uA6EF\uA717-\uA71F\uA722-\uA788\uA78B-\uA7BF\uA7C2-\uA7CA\uA7F5-\uA801\uA803-\uA805\uA807-\uA80A\uA80C-\uA822\uA830-\uA835\uA840-\uA873\uA882-\uA8B3\uA8D0-\uA8D9\uA8F2-\uA8F7\uA8FB\uA8FD\uA8FE\uA900-\uA925\uA930-\uA946\uA960-\uA97C\uA984-\uA9B2\uA9CF-\uA9D9\uA9E0-\uA9E4\uA9E6-\uA9FE\uAA00-\uAA28\uAA40-\uAA42\uAA44-\uAA4B\uAA50-\uAA59\uAA60-\uAA76\uAA7A\uAA7E-\uAAAF\uAAB1\uAAB5\uAAB6\uAAB9-\uAABD\uAAC0\uAAC2\uAADB-\uAADD\uAAE0-\uAAEA\uAAF2-\uAAF4\uAB01-\uAB06\uAB09-\uAB0E\uAB11-\uAB16\uAB20-\uAB26\uAB28-\uAB2E\uAB30-\uAB5A\uAB5C-\uAB69\uAB70-\uABE2\uABF0-\uABF9\uAC00-\uD7A3\uD7B0-\uD7C6\uD7CB-\uD7FB\uF900-\uFA6D\uFA70-\uFAD9\uFB00-\uFB06\uFB13-\uFB17\uFB1D\uFB1F-\uFB28\uFB2A-\uFB36\uFB38-\uFB3C\uFB3E\uFB40\uFB41\uFB43\uFB44\uFB46-\uFBB1\uFBD3-\uFD3D\uFD50-\uFD8F\uFD92-\uFDC7\uFDF0-\uFDFB\uFE70-\uFE74\uFE76-\uFEFC\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A\uFF66-\uFFBE\uFFC2-\uFFC7\uFFCA-\uFFCF\uFFD2-\uFFD7\uFFDA-\uFFDC]|\uD800[\uDC00-\uDC0B\uDC0D-\uDC26\uDC28-\uDC3A\uDC3C\uDC3D\uDC3F-\uDC4D\uDC50-\uDC5D\uDC80-\uDCFA\uDD07-\uDD33\uDD40-\uDD78\uDD8A\uDD8B\uDE80-\uDE9C\uDEA0-\uDED0\uDEE1-\uDEFB\uDF00-\uDF23\uDF2D-\uDF4A\uDF50-\uDF75\uDF80-\uDF9D\uDFA0-\uDFC3\uDFC8-\uDFCF\uDFD1-\uDFD5]|\uD801[\uDC00-\uDC9D\uDCA0-\uDCA9\uDCB0-\uDCD3\uDCD8-\uDCFB\uDD00-\uDD27\uDD30-\uDD63\uDE00-\uDF36\uDF40-\uDF55\uDF60-\uDF67]|\uD802[\uDC00-\uDC05\uDC08\uDC0A-\uDC35\uDC37\uDC38\uDC3C\uDC3F-\uDC55\uDC58-\uDC76\uDC79-\uDC9E\uDCA7-\uDCAF\uDCE0-\uDCF2\uDCF4\uDCF5\uDCFB-\uDD1B\uDD20-\uDD39\uDD80-\uDDB7\uDDBC-\uDDCF\uDDD2-\uDE00\uDE10-\uDE13\uDE15-\uDE17\uDE19-\uDE35\uDE40-\uDE48\uDE60-\uDE7E\uDE80-\uDE9F\uDEC0-\uDEC7\uDEC9-\uDEE4\uDEEB-\uDEEF\uDF00-\uDF35\uDF40-\uDF55\uDF58-\uDF72\uDF78-\uDF91\uDFA9-\uDFAF]|\uD803[\uDC00-\uDC48\uDC80-\uDCB2\uDCC0-\uDCF2\uDCFA-\uDD23\uDD30-\uDD39\uDE60-\uDE7E\uDE80-\uDEA9\uDEB0\uDEB1\uDF00-\uDF27\uDF30-\uDF45\uDF51-\uDF54\uDFB0-\uDFCB\uDFE0-\uDFF6]|\uD804[\uDC03-\uDC37\uDC52-\uDC6F\uDC83-\uDCAF\uDCD0-\uDCE8\uDCF0-\uDCF9\uDD03-\uDD26\uDD36-\uDD3F\uDD44\uDD47\uDD50-\uDD72\uDD76\uDD83-\uDDB2\uDDC1-\uDDC4\uDDD0-\uDDDA\uDDDC\uDDE1-\uDDF4\uDE00-\uDE11\uDE13-\uDE2B\uDE80-\uDE86\uDE88\uDE8A-\uDE8D\uDE8F-\uDE9D\uDE9F-\uDEA8\uDEB0-\uDEDE\uDEF0-\uDEF9\uDF05-\uDF0C\uDF0F\uDF10\uDF13-\uDF28\uDF2A-\uDF30\uDF32\uDF33\uDF35-\uDF39\uDF3D\uDF50\uDF5D-\uDF61]|\uD805[\uDC00-\uDC34\uDC47-\uDC4A\uDC50-\uDC59\uDC5F-\uDC61\uDC80-\uDCAF\uDCC4\uDCC5\uDCC7\uDCD0-\uDCD9\uDD80-\uDDAE\uDDD8-\uDDDB\uDE00-\uDE2F\uDE44\uDE50-\uDE59\uDE80-\uDEAA\uDEB8\uDEC0-\uDEC9\uDF00-\uDF1A\uDF30-\uDF3B]|\uD806[\uDC00-\uDC2B\uDCA0-\uDCF2\uDCFF-\uDD06\uDD09\uDD0C-\uDD13\uDD15\uDD16\uDD18-\uDD2F\uDD3F\uDD41\uDD50-\uDD59\uDDA0-\uDDA7\uDDAA-\uDDD0\uDDE1\uDDE3\uDE00\uDE0B-\uDE32\uDE3A\uDE50\uDE5C-\uDE89\uDE9D\uDEC0-\uDEF8]|\uD807[\uDC00-\uDC08\uDC0A-\uDC2E\uDC40\uDC50-\uDC6C\uDC72-\uDC8F\uDD00-\uDD06\uDD08\uDD09\uDD0B-\uDD30\uDD46\uDD50-\uDD59\uDD60-\uDD65\uDD67\uDD68\uDD6A-\uDD89\uDD98\uDDA0-\uDDA9\uDEE0-\uDEF2\uDFB0\uDFC0-\uDFD4]|\uD808[\uDC00-\uDF99]|\uD809[\uDC00-\uDC6E\uDC80-\uDD43]|[\uD80C\uD81C-\uD820\uD822\uD840-\uD868\uD86A-\uD86C\uD86F-\uD872\uD874-\uD879\uD880-\uD883][\uDC00-\uDFFF]|\uD80D[\uDC00-\uDC2E]|\uD811[\uDC00-\uDE46]|\uD81A[\uDC00-\uDE38\uDE40-\uDE5E\uDE60-\uDE69\uDED0-\uDEED\uDF00-\uDF2F\uDF40-\uDF43\uDF50-\uDF59\uDF5B-\uDF61\uDF63-\uDF77\uDF7D-\uDF8F]|\uD81B[\uDE40-\uDE96\uDF00-\uDF4A\uDF50\uDF93-\uDF9F\uDFE0\uDFE1\uDFE3]|\uD821[\uDC00-\uDFF7]|\uD823[\uDC00-\uDCD5\uDD00-\uDD08]|\uD82C[\uDC00-\uDD1E\uDD50-\uDD52\uDD64-\uDD67\uDD70-\uDEFB]|\uD82F[\uDC00-\uDC6A\uDC70-\uDC7C\uDC80-\uDC88\uDC90-\uDC99]|\uD834[\uDEE0-\uDEF3\uDF60-\uDF78]|\uD835[\uDC00-\uDC54\uDC56-\uDC9C\uDC9E\uDC9F\uDCA2\uDCA5\uDCA6\uDCA9-\uDCAC\uDCAE-\uDCB9\uDCBB\uDCBD-\uDCC3\uDCC5-\uDD05\uDD07-\uDD0A\uDD0D-\uDD14\uDD16-\uDD1C\uDD1E-\uDD39\uDD3B-\uDD3E\uDD40-\uDD44\uDD46\uDD4A-\uDD50\uDD52-\uDEA5\uDEA8-\uDEC0\uDEC2-\uDEDA\uDEDC-\uDEFA\uDEFC-\uDF14\uDF16-\uDF34\uDF36-\uDF4E\uDF50-\uDF6E\uDF70-\uDF88\uDF8A-\uDFA8\uDFAA-\uDFC2\uDFC4-\uDFCB\uDFCE-\uDFFF]|\uD838[\uDD00-\uDD2C\uDD37-\uDD3D\uDD40-\uDD49\uDD4E\uDEC0-\uDEEB\uDEF0-\uDEF9]|\uD83A[\uDC00-\uDCC4\uDCC7-\uDCCF\uDD00-\uDD43\uDD4B\uDD50-\uDD59]|\uD83B[\uDC71-\uDCAB\uDCAD-\uDCAF\uDCB1-\uDCB4\uDD01-\uDD2D\uDD2F-\uDD3D\uDE00-\uDE03\uDE05-\uDE1F\uDE21\uDE22\uDE24\uDE27\uDE29-\uDE32\uDE34-\uDE37\uDE39\uDE3B\uDE42\uDE47\uDE49\uDE4B\uDE4D-\uDE4F\uDE51\uDE52\uDE54\uDE57\uDE59\uDE5B\uDE5D\uDE5F\uDE61\uDE62\uDE64\uDE67-\uDE6A\uDE6C-\uDE72\uDE74-\uDE77\uDE79-\uDE7C\uDE7E\uDE80-\uDE89\uDE8B-\uDE9B\uDEA1-\uDEA3\uDEA5-\uDEA9\uDEAB-\uDEBB]|\uD83C[\uDD00-\uDD0C]|\uD83E[\uDFF0-\uDFF9]|\uD869[\uDC00-\uDEDD\uDF00-\uDFFF]|\uD86D[\uDC00-\uDF34\uDF40-\uDFFF]|\uD86E[\uDC00-\uDC1D\uDC20-\uDFFF]|\uD873[\uDC00-\uDEA1\uDEB0-\uDFFF]|\uD87A[\uDC00-\uDFE0]|\uD87E[\uDC00-\uDE1D]|\uD884[\uDC00-\uDF4A])/,
                ))
            ) {
              var u = r[1] || r[2] || "";
              if (
                !u ||
                (u && ("" === n || this.rules.inline.punctuation.exec(n)))
              ) {
                var i,
                  o,
                  a = r[0].length - 1,
                  s = a,
                  l = 0,
                  c =
                    "*" === r[0][0]
                      ? this.rules.inline.emStrong.rDelimAst
                      : this.rules.inline.emStrong.rDelimUnd;
                for (
                  c.lastIndex = 0, t = t.slice(-1 * e.length + a);
                  null != (r = c.exec(t));

                )
                  if ((i = r[1] || r[2] || r[3] || r[4] || r[5] || r[6]))
                    if (((o = i.length), r[3] || r[4])) s += o;
                    else if (!((r[5] || r[6]) && a % 3) || (a + o) % 3) {
                      if (!((s -= o) > 0)) {
                        if (
                          ((o = Math.min(o, o + s + l)), Math.min(a, o) % 2)
                        ) {
                          var f = e.slice(1, a + r.index + o);
                          return {
                            type: "em",
                            raw: e.slice(0, a + r.index + o + 1),
                            text: f,
                            tokens: this.lexer.inlineTokens(f, []),
                          };
                        }
                        var p = e.slice(2, a + r.index + o - 1);
                        return {
                          type: "strong",
                          raw: e.slice(0, a + r.index + o + 1),
                          text: p,
                          tokens: this.lexer.inlineTokens(p, []),
                        };
                      }
                    } else l += o;
              }
            }
          },
        },
        {
          key: "codespan",
          value: function (e) {
            var t = this.rules.inline.code.exec(e);
            if (t) {
              var n = t[2].replace(/\n/g, " "),
                r = /[^ ]/.test(n),
                u = /^ /.test(n) && / $/.test(n);
              return (
                r && u && (n = n.substring(1, n.length - 1)),
                (n = xf(n, !0)),
                { type: "codespan", raw: t[0], text: n }
              );
            }
          },
        },
        {
          key: "br",
          value: function (e) {
            var t = this.rules.inline.br.exec(e);
            if (t) return { type: "br", raw: t[0] };
          },
        },
        {
          key: "del",
          value: function (e) {
            var t = this.rules.inline.del.exec(e);
            if (t)
              return {
                type: "del",
                raw: t[0],
                text: t[2],
                tokens: this.lexer.inlineTokens(t[2], []),
              };
          },
        },
        {
          key: "autolink",
          value: function (e, t) {
            var n,
              r,
              u = this.rules.inline.autolink.exec(e);
            if (u)
              return (
                (r =
                  "@" === u[2]
                    ? "mailto:" + (n = xf(this.options.mangle ? t(u[1]) : u[1]))
                    : (n = xf(u[1]))),
                {
                  type: "link",
                  raw: u[0],
                  text: n,
                  href: r,
                  tokens: [{ type: "text", raw: n, text: n }],
                }
              );
          },
        },
        {
          key: "url",
          value: function (e, t) {
            var n;
            if ((n = this.rules.inline.url.exec(e))) {
              var r, u;
              if ("@" === n[2])
                u = "mailto:" + (r = xf(this.options.mangle ? t(n[0]) : n[0]));
              else {
                var i;
                do {
                  (i = n[0]),
                    (n[0] = this.rules.inline._backpedal.exec(n[0])[0]);
                } while (i !== n[0]);
                (r = xf(n[0])), (u = "www." === n[1] ? "http://" + r : r);
              }
              return {
                type: "link",
                raw: n[0],
                text: r,
                href: u,
                tokens: [{ type: "text", raw: r, text: r }],
              };
            }
          },
        },
        {
          key: "inlineText",
          value: function (e, t) {
            var n,
              r = this.rules.inline.text.exec(e);
            if (r)
              return (
                (n = this.lexer.state.inRawBlock
                  ? this.options.sanitize
                    ? this.options.sanitizer
                      ? this.options.sanitizer(r[0])
                      : xf(r[0])
                    : r[0]
                  : xf(this.options.smartypants ? t(r[0]) : r[0])),
                { type: "text", raw: r[0], text: n }
              );
          },
        },
      ]),
      e
    );
  })(),
  qf = {
    newline: /^(?: *(?:\n|$))+/,
    code: /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/,
    fences:
      /^ {0,3}(`{3,}(?=[^`\n]*\n)|~{3,})([^\n]*)\n(?:|([\s\S]*?)\n)(?: {0,3}\1[~`]* *(?=\n|$)|$)/,
    hr: /^ {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)/,
    heading: /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,
    blockquote: /^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/,
    list: /^( {0,3}bull)( [^\n]+?)?(?:\n|$)/,
    html: "^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))",
    def: /^ {0,3}\[(label)\]: *(?:\n *)?<?([^\s>]+)>?(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/,
    table: jf,
    lheading: /^([^\n]+)\n {0,3}(=+|-+) *(?:\n+|$)/,
    _paragraph:
      /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,
    text: /^[^\n]+/,
    _label: /(?!\s*\])(?:\\.|[^\[\]\\])+/,
    _title: /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/,
  };
(qf.def = wf(qf.def)
  .replace("label", qf._label)
  .replace("title", qf._title)
  .getRegex()),
  (qf.bullet = /(?:[*+-]|\d{1,9}[.)])/),
  (qf.listItemStart = wf(/^( *)(bull) */)
    .replace("bull", qf.bullet)
    .getRegex()),
  (qf.list = wf(qf.list)
    .replace(/bull/g, qf.bullet)
    .replace(
      "hr",
      "\\n+(?=\\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$))",
    )
    .replace("def", "\\n+(?=" + qf.def.source + ")")
    .getRegex()),
  (qf._tag =
    "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul"),
  (qf._comment = /<!--(?!-?>)[\s\S]*?(?:-->|$)/),
  (qf.html = wf(qf.html, "i")
    .replace("comment", qf._comment)
    .replace("tag", qf._tag)
    .replace(
      "attribute",
      / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/,
    )
    .getRegex()),
  (qf.paragraph = wf(qf._paragraph)
    .replace("hr", qf.hr)
    .replace("heading", " {0,3}#{1,6} ")
    .replace("|lheading", "")
    .replace("|table", "")
    .replace("blockquote", " {0,3}>")
    .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
    .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
    .replace(
      "html",
      "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
    )
    .replace("tag", qf._tag)
    .getRegex()),
  (qf.blockquote = wf(qf.blockquote)
    .replace("paragraph", qf.paragraph)
    .getRegex()),
  (qf.normal = zf({}, qf)),
  (qf.gfm = zf({}, qf.normal, {
    table:
      "^ *([^\\n ].*\\|.*)\\n {0,3}(?:\\| *)?(:?-+:? *(?:\\| *:?-+:? *)*)(?:\\| *)?(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)",
  })),
  (qf.gfm.table = wf(qf.gfm.table)
    .replace("hr", qf.hr)
    .replace("heading", " {0,3}#{1,6} ")
    .replace("blockquote", " {0,3}>")
    .replace("code", " {4}[^\\n]")
    .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
    .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
    .replace(
      "html",
      "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
    )
    .replace("tag", qf._tag)
    .getRegex()),
  (qf.gfm.paragraph = wf(qf._paragraph)
    .replace("hr", qf.hr)
    .replace("heading", " {0,3}#{1,6} ")
    .replace("|lheading", "")
    .replace("table", qf.gfm.table)
    .replace("blockquote", " {0,3}>")
    .replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n")
    .replace("list", " {0,3}(?:[*+-]|1[.)]) ")
    .replace(
      "html",
      "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)",
    )
    .replace("tag", qf._tag)
    .getRegex()),
  (qf.pedantic = zf({}, qf.normal, {
    html: wf(
      "^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))",
    )
      .replace("comment", qf._comment)
      .replace(
        /tag/g,
        "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b",
      )
      .getRegex(),
    def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
    heading: /^(#{1,6})(.*)(?:\n+|$)/,
    fences: jf,
    paragraph: wf(qf.normal._paragraph)
      .replace("hr", qf.hr)
      .replace("heading", " *#{1,6} *[^\n]")
      .replace("lheading", qf.lheading)
      .replace("blockquote", " {0,3}>")
      .replace("|fences", "")
      .replace("|list", "")
      .replace("|html", "")
      .getRegex(),
  }));
var Zf = {
  escape: /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,
  autolink: /^<(scheme:[^\s\x00-\x1f<>]*|email)>/,
  url: jf,
  tag: "^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>",
  link: /^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/,
  reflink: /^!?\[(label)\]\[(ref)\]/,
  nolink: /^!?\[(ref)\](?:\[\])?/,
  reflinkSearch: "reflink|nolink(?!\\()",
  emStrong: {
    lDelim: /^(?:\*+(?:([punct_])|[^\s*]))|^_+(?:([punct*])|([^\s_]))/,
    rDelimAst:
      /^[^_*]*?\_\_[^_*]*?\*[^_*]*?(?=\_\_)|[punct_](\*+)(?=[\s]|$)|[^punct*_\s](\*+)(?=[punct_\s]|$)|[punct_\s](\*+)(?=[^punct*_\s])|[\s](\*+)(?=[punct_])|[punct_](\*+)(?=[punct_])|[^punct*_\s](\*+)(?=[^punct*_\s])/,
    rDelimUnd:
      /^[^_*]*?\*\*[^_*]*?\_[^_*]*?(?=\*\*)|[punct*](\_+)(?=[\s]|$)|[^punct*_\s](\_+)(?=[punct*\s]|$)|[punct*\s](\_+)(?=[^punct*_\s])|[\s](\_+)(?=[punct*])|[punct*](\_+)(?=[punct*])/,
  },
  code: /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,
  br: /^( {2,}|\\)\n(?!\s*$)/,
  del: jf,
  text: /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,
  punctuation: /^([\spunctuation])/,
};
function Gf(e) {
  return e
    .replace(/---/g, "—")
    .replace(/--/g, "–")
    .replace(/(^|[-\u2014/(\[{"\s])'/g, "$1‘")
    .replace(/'/g, "’")
    .replace(/(^|[-\u2014/(\[{\u2018\s])"/g, "$1“")
    .replace(/"/g, "”")
    .replace(/\.{3}/g, "…");
}
function Hf(e) {
  var t,
    n,
    r = "",
    u = e.length;
  for (t = 0; t < u; t++)
    (n = e.charCodeAt(t)),
      Math.random() > 0.5 && (n = "x" + n.toString(16)),
      (r += "&#" + n + ";");
  return r;
}
(Zf._punctuation = "!\"#$%&'()+\\-.,/:;<=>?@\\[\\]`^{|}~"),
  (Zf.punctuation = wf(Zf.punctuation)
    .replace(/punctuation/g, Zf._punctuation)
    .getRegex()),
  (Zf.blockSkip = /\[[^\]]*?\]\([^\)]*?\)|`[^`]*?`|<[^>]*?>/g),
  (Zf.escapedEmSt = /\\\*|\\_/g),
  (Zf._comment = wf(qf._comment).replace("(?:--\x3e|$)", "--\x3e").getRegex()),
  (Zf.emStrong.lDelim = wf(Zf.emStrong.lDelim)
    .replace(/punct/g, Zf._punctuation)
    .getRegex()),
  (Zf.emStrong.rDelimAst = wf(Zf.emStrong.rDelimAst, "g")
    .replace(/punct/g, Zf._punctuation)
    .getRegex()),
  (Zf.emStrong.rDelimUnd = wf(Zf.emStrong.rDelimUnd, "g")
    .replace(/punct/g, Zf._punctuation)
    .getRegex()),
  (Zf._escapes = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/g),
  (Zf._scheme = /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/),
  (Zf._email =
    /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/),
  (Zf.autolink = wf(Zf.autolink)
    .replace("scheme", Zf._scheme)
    .replace("email", Zf._email)
    .getRegex()),
  (Zf._attribute =
    /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/),
  (Zf.tag = wf(Zf.tag)
    .replace("comment", Zf._comment)
    .replace("attribute", Zf._attribute)
    .getRegex()),
  (Zf._label = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/),
  (Zf._href = /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/),
  (Zf._title = /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/),
  (Zf.link = wf(Zf.link)
    .replace("label", Zf._label)
    .replace("href", Zf._href)
    .replace("title", Zf._title)
    .getRegex()),
  (Zf.reflink = wf(Zf.reflink)
    .replace("label", Zf._label)
    .replace("ref", qf._label)
    .getRegex()),
  (Zf.nolink = wf(Zf.nolink).replace("ref", qf._label).getRegex()),
  (Zf.reflinkSearch = wf(Zf.reflinkSearch, "g")
    .replace("reflink", Zf.reflink)
    .replace("nolink", Zf.nolink)
    .getRegex()),
  (Zf.normal = zf({}, Zf)),
  (Zf.pedantic = zf({}, Zf.normal, {
    strong: {
      start: /^__|\*\*/,
      middle: /^__(?=\S)([\s\S]*?\S)__(?!_)|^\*\*(?=\S)([\s\S]*?\S)\*\*(?!\*)/,
      endAst: /\*\*(?!\*)/g,
      endUnd: /__(?!_)/g,
    },
    em: {
      start: /^_|\*/,
      middle: /^()\*(?=\S)([\s\S]*?\S)\*(?!\*)|^_(?=\S)([\s\S]*?\S)_(?!_)/,
      endAst: /\*(?!\*)/g,
      endUnd: /_(?!_)/g,
    },
    link: wf(/^!?\[(label)\]\((.*?)\)/)
      .replace("label", Zf._label)
      .getRegex(),
    reflink: wf(/^!?\[(label)\]\s*\[([^\]]*)\]/)
      .replace("label", Zf._label)
      .getRegex(),
  })),
  (Zf.gfm = zf({}, Zf.normal, {
    escape: wf(Zf.escape).replace("])", "~|])").getRegex(),
    _extended_email:
      /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/,
    url: /^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/,
    _backpedal:
      /(?:[^?!.,:;*_~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_~)]+(?!$))+/,
    del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
    text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/,
  })),
  (Zf.gfm.url = wf(Zf.gfm.url, "i")
    .replace("email", Zf.gfm._extended_email)
    .getRegex()),
  (Zf.breaks = zf({}, Zf.gfm, {
    br: wf(Zf.br).replace("{2,}", "*").getRegex(),
    text: wf(Zf.gfm.text)
      .replace("\\b_", "\\b_| {2,}\\n")
      .replace(/\{2,\}/g, "*")
      .getRegex(),
  }));
var Qf = (function () {
    function e(n) {
      t(this, e),
        (this.tokens = []),
        (this.tokens.links = Object.create(null)),
        (this.options = n || df),
        (this.options.tokenizer = this.options.tokenizer || new Uf()),
        (this.tokenizer = this.options.tokenizer),
        (this.tokenizer.options = this.options),
        (this.tokenizer.lexer = this),
        (this.inlineQueue = []),
        (this.state = { inLink: !1, inRawBlock: !1, top: !0 });
      var r = { block: qf.normal, inline: Zf.normal };
      this.options.pedantic
        ? ((r.block = qf.pedantic), (r.inline = Zf.pedantic))
        : this.options.gfm &&
          ((r.block = qf.gfm),
          this.options.breaks ? (r.inline = Zf.breaks) : (r.inline = Zf.gfm)),
        (this.tokenizer.rules = r);
    }
    return (
      r(
        e,
        [
          {
            key: "lex",
            value: function (e) {
              var t;
              for (
                e = e.replace(/\r\n|\r/g, "\n").replace(/\t/g, "    "),
                  this.blockTokens(e, this.tokens);
                (t = this.inlineQueue.shift());

              )
                this.inlineTokens(t.src, t.tokens);
              return this.tokens;
            },
          },
          {
            key: "blockTokens",
            value: function (e) {
              var t,
                n,
                r,
                u,
                i = this,
                o =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : [];
              for (this.options.pedantic && (e = e.replace(/^ +$/gm, "")); e; )
                if (
                  !(
                    this.options.extensions &&
                    this.options.extensions.block &&
                    this.options.extensions.block.some(function (n) {
                      return (
                        !!(t = n.call({ lexer: i }, e, o)) &&
                        ((e = e.substring(t.raw.length)), o.push(t), !0)
                      );
                    })
                  )
                )
                  if ((t = this.tokenizer.space(e)))
                    (e = e.substring(t.raw.length)),
                      1 === t.raw.length && o.length > 0
                        ? (o[o.length - 1].raw += "\n")
                        : o.push(t);
                  else if ((t = this.tokenizer.code(e)))
                    (e = e.substring(t.raw.length)),
                      !(n = o[o.length - 1]) ||
                      ("paragraph" !== n.type && "text" !== n.type)
                        ? o.push(t)
                        : ((n.raw += "\n" + t.raw),
                          (n.text += "\n" + t.text),
                          (this.inlineQueue[this.inlineQueue.length - 1].src =
                            n.text));
                  else if ((t = this.tokenizer.fences(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.heading(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.hr(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.blockquote(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.list(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.html(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.def(e)))
                    (e = e.substring(t.raw.length)),
                      !(n = o[o.length - 1]) ||
                      ("paragraph" !== n.type && "text" !== n.type)
                        ? this.tokens.links[t.tag] ||
                          (this.tokens.links[t.tag] = {
                            href: t.href,
                            title: t.title,
                          })
                        : ((n.raw += "\n" + t.raw),
                          (n.text += "\n" + t.raw),
                          (this.inlineQueue[this.inlineQueue.length - 1].src =
                            n.text));
                  else if ((t = this.tokenizer.table(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if ((t = this.tokenizer.lheading(e)))
                    (e = e.substring(t.raw.length)), o.push(t);
                  else if (
                    ((r = e),
                    this.options.extensions &&
                      this.options.extensions.startBlock &&
                      (function () {
                        var t = 1 / 0,
                          n = e.slice(1),
                          u = void 0;
                        i.options.extensions.startBlock.forEach(function (e) {
                          "number" == typeof (u = e.call({ lexer: this }, n)) &&
                            u >= 0 &&
                            (t = Math.min(t, u));
                        }),
                          t < 1 / 0 && t >= 0 && (r = e.substring(0, t + 1));
                      })(),
                    this.state.top && (t = this.tokenizer.paragraph(r)))
                  )
                    (n = o[o.length - 1]),
                      u && "paragraph" === n.type
                        ? ((n.raw += "\n" + t.raw),
                          (n.text += "\n" + t.text),
                          this.inlineQueue.pop(),
                          (this.inlineQueue[this.inlineQueue.length - 1].src =
                            n.text))
                        : o.push(t),
                      (u = r.length !== e.length),
                      (e = e.substring(t.raw.length));
                  else if ((t = this.tokenizer.text(e)))
                    (e = e.substring(t.raw.length)),
                      (n = o[o.length - 1]) && "text" === n.type
                        ? ((n.raw += "\n" + t.raw),
                          (n.text += "\n" + t.text),
                          this.inlineQueue.pop(),
                          (this.inlineQueue[this.inlineQueue.length - 1].src =
                            n.text))
                        : o.push(t);
                  else if (e) {
                    var a = "Infinite loop on byte: " + e.charCodeAt(0);
                    if (this.options.silent) {
                      console.error(a);
                      break;
                    }
                    throw new Error(a);
                  }
              return (this.state.top = !0), o;
            },
          },
          {
            key: "inline",
            value: function (e, t) {
              this.inlineQueue.push({ src: e, tokens: t });
            },
          },
          {
            key: "inlineTokens",
            value: function (e) {
              var t,
                n,
                r,
                u,
                i,
                o,
                a = this,
                s =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : [],
                l = e;
              if (this.tokens.links) {
                var c = Object.keys(this.tokens.links);
                if (c.length > 0)
                  for (
                    ;
                    null !=
                    (u = this.tokenizer.rules.inline.reflinkSearch.exec(l));

                  )
                    c.includes(u[0].slice(u[0].lastIndexOf("[") + 1, -1)) &&
                      (l =
                        l.slice(0, u.index) +
                        "[" +
                        Mf("a", u[0].length - 2) +
                        "]" +
                        l.slice(
                          this.tokenizer.rules.inline.reflinkSearch.lastIndex,
                        ));
              }
              for (
                ;
                null != (u = this.tokenizer.rules.inline.blockSkip.exec(l));

              )
                l =
                  l.slice(0, u.index) +
                  "[" +
                  Mf("a", u[0].length - 2) +
                  "]" +
                  l.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
              for (
                ;
                null != (u = this.tokenizer.rules.inline.escapedEmSt.exec(l));

              )
                l =
                  l.slice(0, u.index) +
                  "++" +
                  l.slice(this.tokenizer.rules.inline.escapedEmSt.lastIndex);
              for (; e; )
                if (
                  (i || (o = ""),
                  (i = !1),
                  !(
                    this.options.extensions &&
                    this.options.extensions.inline &&
                    this.options.extensions.inline.some(function (n) {
                      return (
                        !!(t = n.call({ lexer: a }, e, s)) &&
                        ((e = e.substring(t.raw.length)), s.push(t), !0)
                      );
                    })
                  ))
                )
                  if ((t = this.tokenizer.escape(e)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.tag(e)))
                    (e = e.substring(t.raw.length)),
                      (n = s[s.length - 1]) &&
                      "text" === t.type &&
                      "text" === n.type
                        ? ((n.raw += t.raw), (n.text += t.text))
                        : s.push(t);
                  else if ((t = this.tokenizer.link(e)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.reflink(e, this.tokens.links)))
                    (e = e.substring(t.raw.length)),
                      (n = s[s.length - 1]) &&
                      "text" === t.type &&
                      "text" === n.type
                        ? ((n.raw += t.raw), (n.text += t.text))
                        : s.push(t);
                  else if ((t = this.tokenizer.emStrong(e, l, o)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.codespan(e)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.br(e)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.del(e)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if ((t = this.tokenizer.autolink(e, Hf)))
                    (e = e.substring(t.raw.length)), s.push(t);
                  else if (
                    this.state.inLink ||
                    !(t = this.tokenizer.url(e, Hf))
                  ) {
                    if (
                      ((r = e),
                      this.options.extensions &&
                        this.options.extensions.startInline &&
                        (function () {
                          var t = 1 / 0,
                            n = e.slice(1),
                            u = void 0;
                          a.options.extensions.startInline.forEach(
                            function (e) {
                              "number" ==
                                typeof (u = e.call({ lexer: this }, n)) &&
                                u >= 0 &&
                                (t = Math.min(t, u));
                            },
                          ),
                            t < 1 / 0 && t >= 0 && (r = e.substring(0, t + 1));
                        })(),
                      (t = this.tokenizer.inlineText(r, Gf)))
                    )
                      (e = e.substring(t.raw.length)),
                        "_" !== t.raw.slice(-1) && (o = t.raw.slice(-1)),
                        (i = !0),
                        (n = s[s.length - 1]) && "text" === n.type
                          ? ((n.raw += t.raw), (n.text += t.text))
                          : s.push(t);
                    else if (e) {
                      var f = "Infinite loop on byte: " + e.charCodeAt(0);
                      if (this.options.silent) {
                        console.error(f);
                        break;
                      }
                      throw new Error(f);
                    }
                  } else (e = e.substring(t.raw.length)), s.push(t);
              return s;
            },
          },
        ],
        [
          {
            key: "rules",
            get: function () {
              return { block: qf, inline: Zf };
            },
          },
          {
            key: "lex",
            value: function (t, n) {
              return new e(n).lex(t);
            },
          },
          {
            key: "lexInline",
            value: function (t, n) {
              return new e(n).inlineTokens(t);
            },
          },
        ],
      ),
      e
    );
  })(),
  Vf = (function () {
    function e(n) {
      t(this, e), (this.options = n || df);
    }
    return (
      r(e, [
        {
          key: "code",
          value: function (e, t, n) {
            var r = (t || "").match(/\S*/)[0];
            if (this.options.highlight) {
              var u = this.options.highlight(e, r);
              null != u && u !== e && ((n = !0), (e = u));
            }
            return (
              (e = e.replace(/\n$/, "") + "\n"),
              r
                ? '<pre><code class="' +
                  this.options.langPrefix +
                  xf(r, !0) +
                  '">' +
                  (n ? e : xf(e, !0)) +
                  "</code></pre>\n"
                : "<pre><code>" + (n ? e : xf(e, !0)) + "</code></pre>\n"
            );
          },
        },
        {
          key: "blockquote",
          value: function (e) {
            return "<blockquote>\n" + e + "</blockquote>\n";
          },
        },
        {
          key: "html",
          value: function (e) {
            return e;
          },
        },
        {
          key: "heading",
          value: function (e, t, n, r) {
            return this.options.headerIds
              ? "<h" +
                  t +
                  ' id="' +
                  this.options.headerPrefix +
                  r.slug(n) +
                  '">' +
                  e +
                  "</h" +
                  t +
                  ">\n"
              : "<h" + t + ">" + e + "</h" + t + ">\n";
          },
        },
        {
          key: "hr",
          value: function () {
            return this.options.xhtml ? "<hr/>\n" : "<hr>\n";
          },
        },
        {
          key: "list",
          value: function (e, t, n) {
            var r = t ? "ol" : "ul";
            return (
              "<" +
              r +
              (t && 1 !== n ? ' start="' + n + '"' : "") +
              ">\n" +
              e +
              "</" +
              r +
              ">\n"
            );
          },
        },
        {
          key: "listitem",
          value: function (e) {
            return "<li>" + e + "</li>\n";
          },
        },
        {
          key: "checkbox",
          value: function (e) {
            return (
              "<input " +
              (e ? 'checked="" ' : "") +
              'disabled="" type="checkbox"' +
              (this.options.xhtml ? " /" : "") +
              "> "
            );
          },
        },
        {
          key: "paragraph",
          value: function (e) {
            return "<p>" + e + "</p>\n";
          },
        },
        {
          key: "table",
          value: function (e, t) {
            return (
              t && (t = "<tbody>" + t + "</tbody>"),
              "<table>\n<thead>\n" + e + "</thead>\n" + t + "</table>\n"
            );
          },
        },
        {
          key: "tablerow",
          value: function (e) {
            return "<tr>\n" + e + "</tr>\n";
          },
        },
        {
          key: "tablecell",
          value: function (e, t) {
            var n = t.header ? "th" : "td";
            return (
              (t.align
                ? "<" + n + ' align="' + t.align + '">'
                : "<" + n + ">") +
              e +
              "</" +
              n +
              ">\n"
            );
          },
        },
        {
          key: "strong",
          value: function (e) {
            return "<strong>" + e + "</strong>";
          },
        },
        {
          key: "em",
          value: function (e) {
            return "<em>" + e + "</em>";
          },
        },
        {
          key: "codespan",
          value: function (e) {
            return "<code>" + e + "</code>";
          },
        },
        {
          key: "br",
          value: function () {
            return this.options.xhtml ? "<br/>" : "<br>";
          },
        },
        {
          key: "del",
          value: function (e) {
            return "<del>" + e + "</del>";
          },
        },
        {
          key: "link",
          value: function (e, t, n) {
            if (
              null === (e = _f(this.options.sanitize, this.options.baseUrl, e))
            )
              return n;
            var r = '<a href="' + xf(e) + '"';
            return t && (r += ' title="' + t + '"'), (r += ">" + n + "</a>");
          },
        },
        {
          key: "image",
          value: function (e, t, n) {
            if (
              null === (e = _f(this.options.sanitize, this.options.baseUrl, e))
            )
              return n;
            var r = '<img src="' + e + '" alt="' + n + '"';
            return (
              t && (r += ' title="' + t + '"'),
              (r += this.options.xhtml ? "/>" : ">")
            );
          },
        },
        {
          key: "text",
          value: function (e) {
            return e;
          },
        },
      ]),
      e
    );
  })(),
  Yf = (function () {
    function e() {
      t(this, e);
    }
    return (
      r(e, [
        {
          key: "strong",
          value: function (e) {
            return e;
          },
        },
        {
          key: "em",
          value: function (e) {
            return e;
          },
        },
        {
          key: "codespan",
          value: function (e) {
            return e;
          },
        },
        {
          key: "del",
          value: function (e) {
            return e;
          },
        },
        {
          key: "html",
          value: function (e) {
            return e;
          },
        },
        {
          key: "text",
          value: function (e) {
            return e;
          },
        },
        {
          key: "link",
          value: function (e, t, n) {
            return "" + n;
          },
        },
        {
          key: "image",
          value: function (e, t, n) {
            return "" + n;
          },
        },
        {
          key: "br",
          value: function () {
            return "";
          },
        },
      ]),
      e
    );
  })(),
  Kf = (function () {
    function e() {
      t(this, e), (this.seen = {});
    }
    return (
      r(e, [
        {
          key: "serialize",
          value: function (e) {
            return e
              .toLowerCase()
              .trim()
              .replace(/<[!\/a-z].*?>/gi, "")
              .replace(
                /[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,./:;<=>?@[\]^`{|}~]/g,
                "",
              )
              .replace(/\s/g, "-");
          },
        },
        {
          key: "getNextSafeSlug",
          value: function (e, t) {
            var n = e,
              r = 0;
            if (this.seen.hasOwnProperty(n)) {
              r = this.seen[e];
              do {
                n = e + "-" + ++r;
              } while (this.seen.hasOwnProperty(n));
            }
            return t || ((this.seen[e] = r), (this.seen[n] = 0)), n;
          },
        },
        {
          key: "slug",
          value: function (e) {
            var t =
                arguments.length > 1 && void 0 !== arguments[1]
                  ? arguments[1]
                  : {},
              n = this.serialize(e);
            return this.getNextSafeSlug(n, t.dryrun);
          },
        },
      ]),
      e
    );
  })(),
  Xf = (function () {
    function e(n) {
      t(this, e),
        (this.options = n || df),
        (this.options.renderer = this.options.renderer || new Vf()),
        (this.renderer = this.options.renderer),
        (this.renderer.options = this.options),
        (this.textRenderer = new Yf()),
        (this.slugger = new Kf());
    }
    return (
      r(
        e,
        [
          {
            key: "parse",
            value: function (e) {
              var t,
                n,
                r,
                u,
                i,
                o,
                a,
                s,
                l,
                c,
                f,
                p,
                h,
                D,
                g,
                d,
                v,
                y,
                A,
                m =
                  !(arguments.length > 1 && void 0 !== arguments[1]) ||
                  arguments[1],
                k = "",
                E = e.length;
              for (t = 0; t < E; t++)
                if (
                  ((c = e[t]),
                  !(
                    this.options.extensions &&
                    this.options.extensions.renderers &&
                    this.options.extensions.renderers[c.type]
                  ) ||
                    (!1 ===
                      (A = this.options.extensions.renderers[c.type].call(
                        { parser: this },
                        c,
                      )) &&
                      [
                        "space",
                        "hr",
                        "heading",
                        "code",
                        "table",
                        "blockquote",
                        "list",
                        "html",
                        "paragraph",
                        "text",
                      ].includes(c.type)))
                )
                  switch (c.type) {
                    case "space":
                      continue;
                    case "hr":
                      k += this.renderer.hr();
                      continue;
                    case "heading":
                      k += this.renderer.heading(
                        this.parseInline(c.tokens),
                        c.depth,
                        bf(this.parseInline(c.tokens, this.textRenderer)),
                        this.slugger,
                      );
                      continue;
                    case "code":
                      k += this.renderer.code(c.text, c.lang, c.escaped);
                      continue;
                    case "table":
                      for (
                        s = "", a = "", u = c.header.length, n = 0;
                        n < u;
                        n++
                      )
                        a += this.renderer.tablecell(
                          this.parseInline(c.header[n].tokens),
                          { header: !0, align: c.align[n] },
                        );
                      for (
                        s += this.renderer.tablerow(a),
                          l = "",
                          u = c.rows.length,
                          n = 0;
                        n < u;
                        n++
                      ) {
                        for (
                          a = "", i = (o = c.rows[n]).length, r = 0;
                          r < i;
                          r++
                        )
                          a += this.renderer.tablecell(
                            this.parseInline(o[r].tokens),
                            { header: !1, align: c.align[r] },
                          );
                        l += this.renderer.tablerow(a);
                      }
                      k += this.renderer.table(s, l);
                      continue;
                    case "blockquote":
                      (l = this.parse(c.tokens)),
                        (k += this.renderer.blockquote(l));
                      continue;
                    case "list":
                      for (
                        f = c.ordered,
                          p = c.start,
                          h = c.loose,
                          u = c.items.length,
                          l = "",
                          n = 0;
                        n < u;
                        n++
                      )
                        (d = (g = c.items[n]).checked),
                          (v = g.task),
                          (D = ""),
                          g.task &&
                            ((y = this.renderer.checkbox(d)),
                            h
                              ? g.tokens.length > 0 &&
                                "paragraph" === g.tokens[0].type
                                ? ((g.tokens[0].text =
                                    y + " " + g.tokens[0].text),
                                  g.tokens[0].tokens &&
                                    g.tokens[0].tokens.length > 0 &&
                                    "text" === g.tokens[0].tokens[0].type &&
                                    (g.tokens[0].tokens[0].text =
                                      y + " " + g.tokens[0].tokens[0].text))
                                : g.tokens.unshift({ type: "text", text: y })
                              : (D += y)),
                          (D += this.parse(g.tokens, h)),
                          (l += this.renderer.listitem(D, v, d));
                      k += this.renderer.list(l, f, p);
                      continue;
                    case "html":
                      k += this.renderer.html(c.text);
                      continue;
                    case "paragraph":
                      k += this.renderer.paragraph(this.parseInline(c.tokens));
                      continue;
                    case "text":
                      for (
                        l = c.tokens ? this.parseInline(c.tokens) : c.text;
                        t + 1 < E && "text" === e[t + 1].type;

                      )
                        l +=
                          "\n" +
                          ((c = e[++t]).tokens
                            ? this.parseInline(c.tokens)
                            : c.text);
                      k += m ? this.renderer.paragraph(l) : l;
                      continue;
                    default:
                      var x = 'Token with "' + c.type + '" type was not found.';
                      if (this.options.silent) return void console.error(x);
                      throw new Error(x);
                  }
                else k += A || "";
              return k;
            },
          },
          {
            key: "parseInline",
            value: function (e, t) {
              t = t || this.renderer;
              var n,
                r,
                u,
                i = "",
                o = e.length;
              for (n = 0; n < o; n++)
                if (
                  ((r = e[n]),
                  !(
                    this.options.extensions &&
                    this.options.extensions.renderers &&
                    this.options.extensions.renderers[r.type]
                  ) ||
                    (!1 ===
                      (u = this.options.extensions.renderers[r.type].call(
                        { parser: this },
                        r,
                      )) &&
                      [
                        "escape",
                        "html",
                        "link",
                        "image",
                        "strong",
                        "em",
                        "codespan",
                        "br",
                        "del",
                        "text",
                      ].includes(r.type)))
                )
                  switch (r.type) {
                    case "escape":
                      i += t.text(r.text);
                      break;
                    case "html":
                      i += t.html(r.text);
                      break;
                    case "link":
                      i += t.link(
                        r.href,
                        r.title,
                        this.parseInline(r.tokens, t),
                      );
                      break;
                    case "image":
                      i += t.image(r.href, r.title, r.text);
                      break;
                    case "strong":
                      i += t.strong(this.parseInline(r.tokens, t));
                      break;
                    case "em":
                      i += t.em(this.parseInline(r.tokens, t));
                      break;
                    case "codespan":
                      i += t.codespan(r.text);
                      break;
                    case "br":
                      i += t.br();
                      break;
                    case "del":
                      i += t.del(this.parseInline(r.tokens, t));
                      break;
                    case "text":
                      i += t.text(r.text);
                      break;
                    default:
                      var a = 'Token with "' + r.type + '" type was not found.';
                      if (this.options.silent) return void console.error(a);
                      throw new Error(a);
                  }
                else i += u || "";
              return i;
            },
          },
        ],
        [
          {
            key: "parse",
            value: function (t, n) {
              return new e(n).parse(t);
            },
          },
          {
            key: "parseInline",
            value: function (t, n) {
              return new e(n).parseInline(t);
            },
          },
        ],
      ),
      e
    );
  })();
function Wf(e, t, n) {
  if (null == e)
    throw new Error("marked(): input parameter is undefined or null");
  if ("string" != typeof e)
    throw new Error(
      "marked(): input parameter is of type " +
        Object.prototype.toString.call(e) +
        ", string expected",
    );
  if (
    ("function" == typeof t && ((n = t), (t = null)),
    Lf((t = zf({}, Wf.defaults, t || {}))),
    n)
  ) {
    var r,
      u = t.highlight;
    try {
      r = Qf.lex(e, t);
    } catch (e) {
      return n(e);
    }
    var i = function (e) {
      var i;
      if (!e)
        try {
          t.walkTokens && Wf.walkTokens(r, t.walkTokens), (i = Xf.parse(r, t));
        } catch (t) {
          e = t;
        }
      return (t.highlight = u), e ? n(e) : n(null, i);
    };
    if (!u || u.length < 3) return i();
    if ((delete t.highlight, !r.length)) return i();
    var o = 0;
    return (
      Wf.walkTokens(r, function (e) {
        "code" === e.type &&
          (o++,
          setTimeout(function () {
            u(e.text, e.lang, function (t, n) {
              if (t) return i(t);
              null != n && n !== e.text && ((e.text = n), (e.escaped = !0)),
                0 === --o && i();
            });
          }, 0));
      }),
      void (0 === o && i())
    );
  }
  try {
    var a = Qf.lex(e, t);
    return t.walkTokens && Wf.walkTokens(a, t.walkTokens), Xf.parse(a, t);
  } catch (e) {
    if (
      ((e.message +=
        "\nPlease report this to https://github.com/markedjs/marked."),
      t.silent)
    )
      return (
        "<p>An error occurred:</p><pre>" + xf(e.message + "", !0) + "</pre>"
      );
    throw e;
  }
}
(Wf.options = Wf.setOptions =
  function (e) {
    var t;
    return zf(Wf.defaults, e), (t = Wf.defaults), (df = t), Wf;
  }),
  (Wf.getDefaults = gf),
  (Wf.defaults = df),
  (Wf.use = function () {
    for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
      t[n] = arguments[n];
    var r,
      u = zf.apply(void 0, [{}].concat(t)),
      i = Wf.defaults.extensions || { renderers: {}, childTokens: {} };
    t.forEach(function (e) {
      if (
        (e.extensions &&
          ((r = !0),
          e.extensions.forEach(function (e) {
            if (!e.name) throw new Error("extension name required");
            if (e.renderer) {
              var t = i.renderers ? i.renderers[e.name] : null;
              i.renderers[e.name] = t
                ? function () {
                    for (
                      var n = arguments.length, r = new Array(n), u = 0;
                      u < n;
                      u++
                    )
                      r[u] = arguments[u];
                    var i = e.renderer.apply(this, r);
                    return !1 === i && (i = t.apply(this, r)), i;
                  }
                : e.renderer;
            }
            if (e.tokenizer) {
              if (!e.level || ("block" !== e.level && "inline" !== e.level))
                throw new Error("extension level must be 'block' or 'inline'");
              i[e.level]
                ? i[e.level].unshift(e.tokenizer)
                : (i[e.level] = [e.tokenizer]),
                e.start &&
                  ("block" === e.level
                    ? i.startBlock
                      ? i.startBlock.push(e.start)
                      : (i.startBlock = [e.start])
                    : "inline" === e.level &&
                      (i.startInline
                        ? i.startInline.push(e.start)
                        : (i.startInline = [e.start])));
            }
            e.childTokens && (i.childTokens[e.name] = e.childTokens);
          })),
        e.renderer &&
          (function () {
            var t = Wf.defaults.renderer || new Vf(),
              n = function (n) {
                var r = t[n];
                t[n] = function () {
                  for (
                    var u = arguments.length, i = new Array(u), o = 0;
                    o < u;
                    o++
                  )
                    i[o] = arguments[o];
                  var a = e.renderer[n].apply(t, i);
                  return !1 === a && (a = r.apply(t, i)), a;
                };
              };
            for (var r in e.renderer) n(r);
            u.renderer = t;
          })(),
        e.tokenizer &&
          (function () {
            var t = Wf.defaults.tokenizer || new Uf(),
              n = function (n) {
                var r = t[n];
                t[n] = function () {
                  for (
                    var u = arguments.length, i = new Array(u), o = 0;
                    o < u;
                    o++
                  )
                    i[o] = arguments[o];
                  var a = e.tokenizer[n].apply(t, i);
                  return !1 === a && (a = r.apply(t, i)), a;
                };
              };
            for (var r in e.tokenizer) n(r);
            u.tokenizer = t;
          })(),
        e.walkTokens)
      ) {
        var t = Wf.defaults.walkTokens;
        u.walkTokens = function (n) {
          e.walkTokens.call(this, n), t && t.call(this, n);
        };
      }
      r && (u.extensions = i), Wf.setOptions(u);
    });
  }),
  (Wf.walkTokens = function (e, t) {
    var n,
      r = l(e);
    try {
      var u = function () {
        var e = n.value;
        switch ((t.call(Wf, e), e.type)) {
          case "table":
            var r,
              u = l(e.header);
            try {
              for (u.s(); !(r = u.n()).done; ) {
                var i = r.value;
                Wf.walkTokens(i.tokens, t);
              }
            } catch (e) {
              u.e(e);
            } finally {
              u.f();
            }
            var o,
              a = l(e.rows);
            try {
              for (a.s(); !(o = a.n()).done; ) {
                var s,
                  c = l(o.value);
                try {
                  for (c.s(); !(s = c.n()).done; ) {
                    var f = s.value;
                    Wf.walkTokens(f.tokens, t);
                  }
                } catch (e) {
                  c.e(e);
                } finally {
                  c.f();
                }
              }
            } catch (e) {
              a.e(e);
            } finally {
              a.f();
            }
            break;
          case "list":
            Wf.walkTokens(e.items, t);
            break;
          default:
            Wf.defaults.extensions &&
            Wf.defaults.extensions.childTokens &&
            Wf.defaults.extensions.childTokens[e.type]
              ? Wf.defaults.extensions.childTokens[e.type].forEach(
                  function (n) {
                    Wf.walkTokens(e[n], t);
                  },
                )
              : e.tokens && Wf.walkTokens(e.tokens, t);
        }
      };
      for (r.s(); !(n = r.n()).done; ) u();
    } catch (e) {
      r.e(e);
    } finally {
      r.f();
    }
  }),
  (Wf.parseInline = function (e, t) {
    if (null == e)
      throw new Error(
        "marked.parseInline(): input parameter is undefined or null",
      );
    if ("string" != typeof e)
      throw new Error(
        "marked.parseInline(): input parameter is of type " +
          Object.prototype.toString.call(e) +
          ", string expected",
      );
    Lf((t = zf({}, Wf.defaults, t || {})));
    try {
      var n = Qf.lexInline(e, t);
      return (
        t.walkTokens && Wf.walkTokens(n, t.walkTokens), Xf.parseInline(n, t)
      );
    } catch (e) {
      if (
        ((e.message +=
          "\nPlease report this to https://github.com/markedjs/marked."),
        t.silent)
      )
        return (
          "<p>An error occurred:</p><pre>" + xf(e.message + "", !0) + "</pre>"
        );
      throw e;
    }
  }),
  (Wf.Parser = Xf),
  (Wf.parser = Xf.parse),
  (Wf.Renderer = Vf),
  (Wf.TextRenderer = Yf),
  (Wf.Lexer = Qf),
  (Wf.lexer = Qf.lex),
  (Wf.Tokenizer = Uf),
  (Wf.Slugger = Kf),
  (Wf.parse = Wf);
var Jf = /\[([\s\d,|-]*)\]/,
  ep = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export default function () {
  var t;
  function n(e) {
    var t = (
        e.querySelector("[data-template]") ||
        e.querySelector("script") ||
        e
      ).textContent,
      n = (t = t.replace(new RegExp("__SCRIPT_END__", "g"), "</script>")).match(
        /^\n?(\s*)/,
      )[1].length,
      r = t.match(/^\n?(\t*)/)[1].length;
    return (
      r > 0
        ? (t = t.replace(new RegExp("\\n?\\t{" + r + "}", "g"), "\n"))
        : n > 1 && (t = t.replace(new RegExp("\\n? {" + n + "}", "g"), "\n")),
      t
    );
  }
  function r(e) {
    for (var t = e.attributes, n = [], r = 0, u = t.length; r < u; r++) {
      var i = t[r].name,
        o = t[r].value;
      /data\-(markdown|separator|vertical|notes)/gi.test(i) ||
        (o ? n.push(i + '="' + o + '"') : n.push(i));
    }
    return n.join(" ");
  }
  function o(e) {
    return (
      ((e = e || {}).separator = e.separator || "\r?\n---\r?\n"),
      (e.notesSeparator = e.notesSeparator || "notes?:"),
      (e.attributes = e.attributes || ""),
      e
    );
  }
  function a(e, t) {
    t = o(t);
    var n = e.split(new RegExp(t.notesSeparator, "mgi"));
    return (
      2 === n.length &&
        (e = n[0] + '<aside class="notes">' + Wf(n[1].trim()) + "</aside>"),
      '<script type="text/template">' +
        (e = e.replace(/<\/script>/g, "__SCRIPT_END__")) +
        "</script>"
    );
  }
  function s(e, t) {
    t = o(t);
    for (
      var n,
        r,
        u,
        i = new RegExp(
          t.separator + (t.verticalSeparator ? "|" + t.verticalSeparator : ""),
          "mg",
        ),
        s = new RegExp(t.separator),
        l = 0,
        c = !0,
        f = [];
      (n = i.exec(e));

    )
      !(r = s.test(n[0])) && c && f.push([]),
        (u = e.substring(l, n.index)),
        r && c ? f.push(u) : f[f.length - 1].push(u),
        (l = i.lastIndex),
        (c = r);
    (c ? f : f[f.length - 1]).push(e.substring(l));
    for (var p = "", h = 0, D = f.length; h < D; h++)
      f[h] instanceof Array
        ? ((p += "<section " + t.attributes + ">"),
          f[h].forEach(function (e) {
            p += "<section data-markdown>" + a(e, t) + "</section>";
          }),
          (p += "</section>"))
        : (p +=
            "<section " +
            t.attributes +
            " data-markdown>" +
            a(f[h], t) +
            "</section>");
    return p;
  }
  function l(e) {
    return new Promise(function (t) {
      var u = [];
      [].slice
        .call(
          e.querySelectorAll(
            "section[data-markdown]:not([data-markdown-parsed])",
          ),
        )
        .forEach(function (e, t) {
          e.getAttribute("data-markdown").length
            ? u.push(
                (function (e) {
                  return new Promise(function (t, n) {
                    var r = new XMLHttpRequest(),
                      u = e.getAttribute("data-markdown"),
                      i = e.getAttribute("data-charset");
                    null != i &&
                      "" != i &&
                      r.overrideMimeType("text/html; charset=" + i),
                      (r.onreadystatechange = function (e, r) {
                        4 === r.readyState &&
                          ((r.status >= 200 && r.status < 300) || 0 === r.status
                            ? t(r, u)
                            : n(r, u));
                      }.bind(this, e, r)),
                      r.open("GET", u, !0);
                    try {
                      r.send();
                    } catch (e) {
                      console.warn(
                        "Failed to get the Markdown file " +
                          u +
                          ". Make sure that the presentation and the file are served by a HTTP server and the file can be found there. " +
                          e,
                      ),
                        t(r, u);
                    }
                  });
                })(e).then(
                  function (t, n) {
                    e.outerHTML = s(t.responseText, {
                      separator: e.getAttribute("data-separator"),
                      verticalSeparator: e.getAttribute(
                        "data-separator-vertical",
                      ),
                      notesSeparator: e.getAttribute("data-separator-notes"),
                      attributes: r(e),
                    });
                  },
                  function (t, n) {
                    e.outerHTML =
                      '<section data-state="alert">ERROR: The attempt to fetch ' +
                      n +
                      " failed with HTTP status " +
                      t.status +
                      ".Check your browser's JavaScript console for more details.<p>Remember that you need to serve the presentation HTML from a HTTP server.</p></section>";
                  },
                ),
              )
            : (e.outerHTML = s(n(e), {
                separator: e.getAttribute("data-separator"),
                verticalSeparator: e.getAttribute("data-separator-vertical"),
                notesSeparator: e.getAttribute("data-separator-notes"),
                attributes: r(e),
              }));
        }),
        Promise.all(u).then(t);
    });
  }
  function c(e, t, n) {
    var r,
      u,
      i = new RegExp(n, "mg"),
      o = new RegExp('([^"= ]+?)="([^"]+?)"|(data-[^"= ]+?)(?=[" ])', "mg"),
      a = e.nodeValue;
    if ((r = i.exec(a))) {
      var s = r[1];
      for (
        a = a.substring(0, r.index) + a.substring(i.lastIndex), e.nodeValue = a;
        (u = o.exec(s));

      )
        u[2] ? t.setAttribute(u[1], u[2]) : t.setAttribute(u[3], "");
      return !0;
    }
    return !1;
  }
  function f(e, t, n, r, u) {
    if (null != t && null != t.childNodes && t.childNodes.length > 0)
      for (var i = t, o = 0; o < t.childNodes.length; o++) {
        var a = t.childNodes[o];
        if (o > 0)
          for (var s = o - 1; s >= 0; ) {
            var l = t.childNodes[s];
            if ("function" == typeof l.setAttribute && "BR" != l.tagName) {
              i = l;
              break;
            }
            s -= 1;
          }
        var p = e;
        "section" == a.nodeName && ((p = a), (i = a)),
          ("function" != typeof a.setAttribute &&
            a.nodeType != Node.COMMENT_NODE) ||
            f(p, a, i, r, u);
      }
    t.nodeType == Node.COMMENT_NODE && 0 == c(t, n, r) && c(t, e, u);
  }
  function p() {
    var e = t
      .getRevealElement()
      .querySelectorAll("[data-markdown]:not([data-markdown-parsed])");
    return (
      [].slice.call(e).forEach(function (e) {
        e.setAttribute("data-markdown-parsed", !0);
        var t = e.querySelector("aside.notes"),
          r = n(e);
        (e.innerHTML = Wf(r)),
          f(
            e,
            e,
            null,
            e.getAttribute("data-element-attributes") ||
              e.parentNode.getAttribute("data-element-attributes") ||
              "\\.element\\s*?(.+?)$",
            e.getAttribute("data-attributes") ||
              e.parentNode.getAttribute("data-attributes") ||
              "\\.slide:\\s*?(\\S.+?)$",
          ),
          t && e.appendChild(t);
      }),
      Promise.resolve()
    );
  }
  return {
    id: "markdown",
    init: function (n) {
      var r = (t = n).getConfig().markdown || {},
        o = r.renderer,
        a = r.animateLists,
        s = i(r, ["renderer", "animateLists"]);
      return (
        o ||
          ((o = new Wf.Renderer()).code = function (e, t) {
            var n = "";
            return (
              Jf.test(t) &&
                ((n = t.match(Jf)[1].trim()),
                (n = 'data-line-numbers="'.concat(n, '"')),
                (t = t.replace(Jf, "").trim())),
              (e = e.replace(/([&<>'"])/g, function (e) {
                return ep[e];
              })),
              "<pre><code "
                .concat(n, ' class="')
                .concat(t, '">')
                .concat(e, "</code></pre>")
            );
          }),
        !0 === a &&
          (o.listitem = function (e) {
            return '<li class="fragment">'.concat(e, "</li>");
          }),
        Wf.setOptions(
          (function (t) {
            for (var n = 1; n < arguments.length; n++) {
              var r = null != arguments[n] ? arguments[n] : {};
              n % 2
                ? e(Object(r), !0).forEach(function (e) {
                    u(t, e, r[e]);
                  })
                : Object.getOwnPropertyDescriptors
                  ? Object.defineProperties(
                      t,
                      Object.getOwnPropertyDescriptors(r),
                    )
                  : e(Object(r)).forEach(function (e) {
                      Object.defineProperty(
                        t,
                        e,
                        Object.getOwnPropertyDescriptor(r, e),
                      );
                    });
            }
            return t;
          })({ renderer: o }, s),
        ),
        l(t.getRevealElement()).then(p)
      );
    },
    processSlides: l,
    convertSlides: p,
    slidify: s,
    marked: Wf,
  };
}