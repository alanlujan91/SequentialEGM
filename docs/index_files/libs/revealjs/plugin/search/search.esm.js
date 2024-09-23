var e =
    "undefined" != typeof globalThis
      ? globalThis
      : "undefined" != typeof window
        ? window
        : "undefined" != typeof global
          ? global
          : "undefined" != typeof self
            ? self
            : {},
  t = function (e) {
    try {
      return !!e();
    } catch (e) {
      return !0;
    }
  },
  n = !t(function () {
    return (
      7 !=
      Object.defineProperty({}, 1, {
        get: function () {
          return 7;
        },
      })[1]
    );
  }),
  r = function (e) {
    return e && e.Math == Math && e;
  },
  o =
    r("object" == typeof globalThis && globalThis) ||
    r("object" == typeof window && window) ||
    r("object" == typeof self && self) ||
    r("object" == typeof e && e) ||
    (function () {
      return this;
    })() ||
    Function("return this")(),
  i = t,
  c = /#|\.prototype\./,
  a = function (e, t) {
    var n = l[u(e)];
    return n == s || (n != f && ("function" == typeof t ? i(t) : !!t));
  },
  u = (a.normalize = function (e) {
    return String(e).replace(c, ".").toLowerCase();
  }),
  l = (a.data = {}),
  f = (a.NATIVE = "N"),
  s = (a.POLYFILL = "P"),
  p = a,
  g = function (e) {
    return "object" == typeof e ? null !== e : "function" == typeof e;
  },
  d = g,
  h = function (e) {
    if (!d(e)) throw TypeError(String(e) + " is not an object");
    return e;
  },
  y = g,
  v = h,
  x = function (e) {
    if (!y(e) && null !== e)
      throw TypeError("Can't set " + String(e) + " as a prototype");
    return e;
  },
  b =
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
            return v(n), x(r), t ? e.call(n, r) : (n.__proto__ = r), n;
          };
        })()
      : void 0),
  E = g,
  m = b,
  S = {},
  w = g,
  O = o.document,
  R = w(O) && w(O.createElement),
  T = function (e) {
    return R ? O.createElement(e) : {};
  },
  _ =
    !n &&
    !t(function () {
      return (
        7 !=
        Object.defineProperty(T("div"), "a", {
          get: function () {
            return 7;
          },
        }).a
      );
    }),
  j = g,
  P = function (e, t) {
    if (!j(e)) return e;
    var n, r;
    if (t && "function" == typeof (n = e.toString) && !j((r = n.call(e))))
      return r;
    if ("function" == typeof (n = e.valueOf) && !j((r = n.call(e)))) return r;
    if (!t && "function" == typeof (n = e.toString) && !j((r = n.call(e))))
      return r;
    throw TypeError("Can't convert object to primitive value");
  },
  I = n,
  C = _,
  N = h,
  A = P,
  k = Object.defineProperty;
S.f = I
  ? k
  : function (e, t, n) {
      if ((N(e), (t = A(t, !0)), N(n), C))
        try {
          return k(e, t, n);
        } catch (e) {}
      if ("get" in n || "set" in n) throw TypeError("Accessors not supported");
      return "value" in n && (e[t] = n.value), e;
    };
var $ = {},
  L = function (e) {
    if (null == e) throw TypeError("Can't call method on " + e);
    return e;
  },
  M = L,
  U = function (e) {
    return Object(M(e));
  },
  D = U,
  F = {}.hasOwnProperty,
  z = function (e, t) {
    return F.call(D(e), t);
  },
  K = {}.toString,
  B = function (e) {
    return K.call(e).slice(8, -1);
  },
  W = B,
  G = "".split,
  V = t(function () {
    return !Object("z").propertyIsEnumerable(0);
  })
    ? function (e) {
        return "String" == W(e) ? G.call(e, "") : Object(e);
      }
    : Object,
  Y = L,
  q = function (e) {
    return V(Y(e));
  },
  X = Math.ceil,
  H = Math.floor,
  J = function (e) {
    return isNaN((e = +e)) ? 0 : (e > 0 ? H : X)(e);
  },
  Q = J,
  Z = Math.min,
  ee = function (e) {
    return e > 0 ? Z(Q(e), 9007199254740991) : 0;
  },
  te = J,
  ne = Math.max,
  re = Math.min,
  oe = q,
  ie = ee,
  ce = function (e, t) {
    var n = te(e);
    return n < 0 ? ne(n + t, 0) : re(n, t);
  },
  ae = function (e) {
    return function (t, n, r) {
      var o,
        i = oe(t),
        c = ie(i.length),
        a = ce(r, c);
      if (e && n != n) {
        for (; c > a; ) if ((o = i[a++]) != o) return !0;
      } else
        for (; c > a; a++) if ((e || a in i) && i[a] === n) return e || a || 0;
      return !e && -1;
    };
  },
  ue = { includes: ae(!0), indexOf: ae(!1) },
  le = {},
  fe = z,
  se = q,
  pe = ue.indexOf,
  ge = le,
  de = function (e, t) {
    var n,
      r = se(e),
      o = 0,
      i = [];
    for (n in r) !fe(ge, n) && fe(r, n) && i.push(n);
    for (; t.length > o; ) fe(r, (n = t[o++])) && (~pe(i, n) || i.push(n));
    return i;
  },
  he = [
    "constructor",
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toLocaleString",
    "toString",
    "valueOf",
  ].concat("length", "prototype");
$.f =
  Object.getOwnPropertyNames ||
  function (e) {
    return de(e, he);
  };
var ye = { exports: {} },
  ve = function (e, t) {
    return {
      enumerable: !(1 & e),
      configurable: !(2 & e),
      writable: !(4 & e),
      value: t,
    };
  },
  xe = S,
  be = ve,
  Ee = n
    ? function (e, t, n) {
        return xe.f(e, t, be(1, n));
      }
    : function (e, t, n) {
        return (e[t] = n), e;
      },
  me = o,
  Se = Ee,
  we = function (e, t) {
    try {
      Se(me, e, t);
    } catch (n) {
      me[e] = t;
    }
    return t;
  },
  Oe = we,
  Re = o["__core-js_shared__"] || Oe("__core-js_shared__", {}),
  Te = Re;
(ye.exports = function (e, t) {
  return Te[e] || (Te[e] = void 0 !== t ? t : {});
})("versions", []).push({
  version: "3.12.1",
  mode: "global",
  copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
});
var _e,
  je,
  Pe = 0,
  Ie = Math.random(),
  Ce = function (e) {
    return (
      "Symbol(" +
      String(void 0 === e ? "" : e) +
      ")_" +
      (++Pe + Ie).toString(36)
    );
  },
  Ne = o,
  Ae = o,
  ke = function (e) {
    return "function" == typeof e ? e : void 0;
  },
  $e = function (e, t) {
    return arguments.length < 2
      ? ke(Ne[e]) || ke(Ae[e])
      : (Ne[e] && Ne[e][t]) || (Ae[e] && Ae[e][t]);
  },
  Le = $e("navigator", "userAgent") || "",
  Me = o.process,
  Ue = Me && Me.versions,
  De = Ue && Ue.v8;
De
  ? (je = (_e = De.split("."))[0] < 4 ? 1 : _e[0] + _e[1])
  : Le &&
    (!(_e = Le.match(/Edge\/(\d+)/)) || _e[1] >= 74) &&
    (_e = Le.match(/Chrome\/(\d+)/)) &&
    (je = _e[1]);
var Fe = je && +je,
  ze = t,
  Ke =
    !!Object.getOwnPropertySymbols &&
    !ze(function () {
      return !String(Symbol()) || (!Symbol.sham && Fe && Fe < 41);
    }),
  Be = Ke && !Symbol.sham && "symbol" == typeof Symbol.iterator,
  We = o,
  Ge = ye.exports,
  Ve = z,
  Ye = Ce,
  qe = Ke,
  Xe = Be,
  He = Ge("wks"),
  Je = We.Symbol,
  Qe = Xe ? Je : (Je && Je.withoutSetter) || Ye,
  Ze = function (e) {
    return (
      (Ve(He, e) && (qe || "string" == typeof He[e])) ||
        (qe && Ve(Je, e) ? (He[e] = Je[e]) : (He[e] = Qe("Symbol." + e))),
      He[e]
    );
  },
  et = g,
  tt = B,
  nt = Ze("match"),
  rt = h,
  ot = function () {
    var e = rt(this),
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
  it = {},
  ct = t;
function at(e, t) {
  return RegExp(e, t);
}
(it.UNSUPPORTED_Y = ct(function () {
  var e = at("a", "y");
  return (e.lastIndex = 2), null != e.exec("abcd");
})),
  (it.BROKEN_CARET = ct(function () {
    var e = at("^r", "gy");
    return (e.lastIndex = 2), null != e.exec("str");
  }));
var ut = { exports: {} },
  lt = Re,
  ft = Function.toString;
"function" != typeof lt.inspectSource &&
  (lt.inspectSource = function (e) {
    return ft.call(e);
  });
var st,
  pt,
  gt,
  dt = lt.inspectSource,
  ht = dt,
  yt = o.WeakMap,
  vt = "function" == typeof yt && /native code/.test(ht(yt)),
  xt = ye.exports,
  bt = Ce,
  Et = xt("keys"),
  mt = vt,
  St = g,
  wt = Ee,
  Ot = z,
  Rt = Re,
  Tt = function (e) {
    return Et[e] || (Et[e] = bt(e));
  },
  _t = le,
  jt = o.WeakMap;
if (mt || Rt.state) {
  var Pt = Rt.state || (Rt.state = new jt()),
    It = Pt.get,
    Ct = Pt.has,
    Nt = Pt.set;
  (st = function (e, t) {
    if (Ct.call(Pt, e)) throw new TypeError("Object already initialized");
    return (t.facade = e), Nt.call(Pt, e, t), t;
  }),
    (pt = function (e) {
      return It.call(Pt, e) || {};
    }),
    (gt = function (e) {
      return Ct.call(Pt, e);
    });
} else {
  var At = Tt("state");
  (_t[At] = !0),
    (st = function (e, t) {
      if (Ot(e, At)) throw new TypeError("Object already initialized");
      return (t.facade = e), wt(e, At, t), t;
    }),
    (pt = function (e) {
      return Ot(e, At) ? e[At] : {};
    }),
    (gt = function (e) {
      return Ot(e, At);
    });
}
var kt = {
    set: st,
    get: pt,
    has: gt,
    enforce: function (e) {
      return gt(e) ? pt(e) : st(e, {});
    },
    getterFor: function (e) {
      return function (t) {
        var n;
        if (!St(t) || (n = pt(t)).type !== e)
          throw TypeError("Incompatible receiver, " + e + " required");
        return n;
      };
    },
  },
  $t = o,
  Lt = Ee,
  Mt = z,
  Ut = we,
  Dt = dt,
  Ft = kt.get,
  zt = kt.enforce,
  Kt = String(String).split("String");
(ut.exports = function (e, t, n, r) {
  var o,
    i = !!r && !!r.unsafe,
    c = !!r && !!r.enumerable,
    a = !!r && !!r.noTargetGet;
  "function" == typeof n &&
    ("string" != typeof t || Mt(n, "name") || Lt(n, "name", t),
    (o = zt(n)).source || (o.source = Kt.join("string" == typeof t ? t : ""))),
    e !== $t
      ? (i ? !a && e[t] && (c = !0) : delete e[t], c ? (e[t] = n) : Lt(e, t, n))
      : c
        ? (e[t] = n)
        : Ut(t, n);
})(Function.prototype, "toString", function () {
  return ("function" == typeof this && Ft(this).source) || Dt(this);
});
var Bt = $e,
  Wt = S,
  Gt = n,
  Vt = Ze("species"),
  Yt = n,
  qt = o,
  Xt = p,
  Ht = function (e, t, n) {
    var r, o;
    return (
      m &&
        "function" == typeof (r = t.constructor) &&
        r !== n &&
        E((o = r.prototype)) &&
        o !== n.prototype &&
        m(e, o),
      e
    );
  },
  Jt = S.f,
  Qt = $.f,
  Zt = function (e) {
    var t;
    return et(e) && (void 0 !== (t = e[nt]) ? !!t : "RegExp" == tt(e));
  },
  en = ot,
  tn = it,
  nn = ut.exports,
  rn = t,
  on = kt.enforce,
  cn = function (e) {
    var t = Bt(e),
      n = Wt.f;
    Gt &&
      t &&
      !t[Vt] &&
      n(t, Vt, {
        configurable: !0,
        get: function () {
          return this;
        },
      });
  },
  an = Ze("match"),
  un = qt.RegExp,
  ln = un.prototype,
  fn = /a/g,
  sn = /a/g,
  pn = new un(fn) !== fn,
  gn = tn.UNSUPPORTED_Y;
if (
  Yt &&
  Xt(
    "RegExp",
    !pn ||
      gn ||
      rn(function () {
        return (
          (sn[an] = !1), un(fn) != fn || un(sn) == sn || "/a/i" != un(fn, "i")
        );
      }),
  )
) {
  for (
    var dn = function (e, t) {
        var n,
          r = this instanceof dn,
          o = Zt(e),
          i = void 0 === t;
        if (!r && o && e.constructor === dn && i) return e;
        pn
          ? o && !i && (e = e.source)
          : e instanceof dn && (i && (t = en.call(e)), (e = e.source)),
          gn && (n = !!t && t.indexOf("y") > -1) && (t = t.replace(/y/g, ""));
        var c = Ht(pn ? new un(e, t) : un(e, t), r ? this : ln, dn);
        gn && n && (on(c).sticky = !0);
        return c;
      },
      hn = function (e) {
        (e in dn) ||
          Jt(dn, e, {
            configurable: !0,
            get: function () {
              return un[e];
            },
            set: function (t) {
              un[e] = t;
            },
          });
      },
      yn = Qt(un),
      vn = 0;
    yn.length > vn;

  )
    hn(yn[vn++]);
  (ln.constructor = dn), (dn.prototype = ln), nn(qt, "RegExp", dn);
}
cn("RegExp");
var xn = {},
  bn = {},
  En = {}.propertyIsEnumerable,
  mn = Object.getOwnPropertyDescriptor,
  Sn = mn && !En.call({ 1: 2 }, 1);
bn.f = Sn
  ? function (e) {
      var t = mn(this, e);
      return !!t && t.enumerable;
    }
  : En;
var wn = n,
  On = bn,
  Rn = ve,
  Tn = q,
  _n = P,
  jn = z,
  Pn = _,
  In = Object.getOwnPropertyDescriptor;
xn.f = wn
  ? In
  : function (e, t) {
      if (((e = Tn(e)), (t = _n(t, !0)), Pn))
        try {
          return In(e, t);
        } catch (e) {}
      if (jn(e, t)) return Rn(!On.f.call(e, t), e[t]);
    };
var Cn = {};
Cn.f = Object.getOwnPropertySymbols;
var Nn = $,
  An = Cn,
  kn = h,
  $n =
    $e("Reflect", "ownKeys") ||
    function (e) {
      var t = Nn.f(kn(e)),
        n = An.f;
      return n ? t.concat(n(e)) : t;
    },
  Ln = z,
  Mn = $n,
  Un = xn,
  Dn = S,
  Fn = o,
  zn = xn.f,
  Kn = Ee,
  Bn = ut.exports,
  Wn = we,
  Gn = function (e, t) {
    for (var n = Mn(t), r = Dn.f, o = Un.f, i = 0; i < n.length; i++) {
      var c = n[i];
      Ln(e, c) || r(e, c, o(t, c));
    }
  },
  Vn = p,
  Yn = ot,
  qn = it,
  Xn = ye.exports,
  Hn = RegExp.prototype.exec,
  Jn = Xn("native-string-replace", String.prototype.replace),
  Qn = Hn,
  Zn = (function () {
    var e = /a/,
      t = /b*/g;
    return (
      Hn.call(e, "a"), Hn.call(t, "a"), 0 !== e.lastIndex || 0 !== t.lastIndex
    );
  })(),
  er = qn.UNSUPPORTED_Y || qn.BROKEN_CARET,
  tr = void 0 !== /()??/.exec("")[1];
(Zn || tr || er) &&
  (Qn = function (e) {
    var t,
      n,
      r,
      o,
      i = this,
      c = er && i.sticky,
      a = Yn.call(i),
      u = i.source,
      l = 0,
      f = e;
    return (
      c &&
        (-1 === (a = a.replace("y", "")).indexOf("g") && (a += "g"),
        (f = String(e).slice(i.lastIndex)),
        i.lastIndex > 0 &&
          (!i.multiline || (i.multiline && "\n" !== e[i.lastIndex - 1])) &&
          ((u = "(?: " + u + ")"), (f = " " + f), l++),
        (n = new RegExp("^(?:" + u + ")", a))),
      tr && (n = new RegExp("^" + u + "$(?!\\s)", a)),
      Zn && (t = i.lastIndex),
      (r = Hn.call(c ? n : i, f)),
      c
        ? r
          ? ((r.input = r.input.slice(l)),
            (r[0] = r[0].slice(l)),
            (r.index = i.lastIndex),
            (i.lastIndex += r[0].length))
          : (i.lastIndex = 0)
        : Zn && r && (i.lastIndex = i.global ? r.index + r[0].length : t),
      tr &&
        r &&
        r.length > 1 &&
        Jn.call(r[0], n, function () {
          for (o = 1; o < arguments.length - 2; o++)
            void 0 === arguments[o] && (r[o] = void 0);
        }),
      r
    );
  });
var nr = Qn;
(function (e, t) {
  var n,
    r,
    o,
    i,
    c,
    a = e.target,
    u = e.global,
    l = e.stat;
  if ((n = u ? Fn : l ? Fn[a] || Wn(a, {}) : (Fn[a] || {}).prototype))
    for (r in t) {
      if (
        ((i = t[r]),
        (o = e.noTargetGet ? (c = zn(n, r)) && c.value : n[r]),
        !Vn(u ? r : a + (l ? "." : "#") + r, e.forced) && void 0 !== o)
      ) {
        if (typeof i == typeof o) continue;
        Gn(i, o);
      }
      (e.sham || (o && o.sham)) && Kn(i, "sham", !0), Bn(n, r, i, e);
    }
})({ target: "RegExp", proto: !0, forced: /./.exec !== nr }, { exec: nr });
var rr = ut.exports,
  or = h,
  ir = t,
  cr = ot,
  ar = RegExp.prototype,
  ur = ar.toString,
  lr = ir(function () {
    return "/a/b" != ur.call({ source: "a", flags: "b" });
  }),
  fr = "toString" != ur.name;
(lr || fr) &&
  rr(
    RegExp.prototype,
    "toString",
    function () {
      var e = or(this),
        t = String(e.source),
        n = e.flags;
      return (
        "/" +
        t +
        "/" +
        String(
          void 0 === n && e instanceof RegExp && !("flags" in ar)
            ? cr.call(e)
            : n,
        )
      );
    },
    { unsafe: !0 },
  );
var sr = ut.exports,
  pr = nr,
  gr = t,
  dr = Ze,
  hr = Ee,
  yr = dr("species"),
  vr = RegExp.prototype,
  xr = !gr(function () {
    var e = /./;
    return (
      (e.exec = function () {
        var e = [];
        return (e.groups = { a: "7" }), e;
      }),
      "7" !== "".replace(e, "$<a>")
    );
  }),
  br = "$0" === "a".replace(/./, "$0"),
  Er = dr("replace"),
  mr = !!/./[Er] && "" === /./[Er]("a", "$0"),
  Sr = !gr(function () {
    var e = /(?:)/,
      t = e.exec;
    e.exec = function () {
      return t.apply(this, arguments);
    };
    var n = "ab".split(e);
    return 2 !== n.length || "a" !== n[0] || "b" !== n[1];
  }),
  wr = J,
  Or = L,
  Rr = function (e) {
    return function (t, n) {
      var r,
        o,
        i = String(Or(t)),
        c = wr(n),
        a = i.length;
      return c < 0 || c >= a
        ? e
          ? ""
          : void 0
        : (r = i.charCodeAt(c)) < 55296 ||
            r > 56319 ||
            c + 1 === a ||
            (o = i.charCodeAt(c + 1)) < 56320 ||
            o > 57343
          ? e
            ? i.charAt(c)
            : r
          : e
            ? i.slice(c, c + 2)
            : o - 56320 + ((r - 55296) << 10) + 65536;
    };
  },
  Tr = { codeAt: Rr(!1), charAt: Rr(!0) }.charAt,
  _r = U,
  jr = Math.floor,
  Pr = "".replace,
  Ir = /\$([$&'`]|\d{1,2}|<[^>]*>)/g,
  Cr = /\$([$&'`]|\d{1,2})/g,
  Nr = B,
  Ar = nr,
  kr = function (e, t, n, r) {
    var o = dr(e),
      i = !gr(function () {
        var t = {};
        return (
          (t[o] = function () {
            return 7;
          }),
          7 != ""[e](t)
        );
      }),
      c =
        i &&
        !gr(function () {
          var t = !1,
            n = /a/;
          return (
            "split" === e &&
              (((n = {}).constructor = {}),
              (n.constructor[yr] = function () {
                return n;
              }),
              (n.flags = ""),
              (n[o] = /./[o])),
            (n.exec = function () {
              return (t = !0), null;
            }),
            n[o](""),
            !t
          );
        });
    if (
      !i ||
      !c ||
      ("replace" === e && (!xr || !br || mr)) ||
      ("split" === e && !Sr)
    ) {
      var a = /./[o],
        u = n(
          o,
          ""[e],
          function (e, t, n, r, o) {
            var c = t.exec;
            return c === pr || c === vr.exec
              ? i && !o
                ? { done: !0, value: a.call(t, n, r) }
                : { done: !0, value: e.call(n, t, r) }
              : { done: !1 };
          },
          {
            REPLACE_KEEPS_$0: br,
            REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE: mr,
          },
        ),
        l = u[0],
        f = u[1];
      sr(String.prototype, e, l),
        sr(
          vr,
          o,
          2 == t
            ? function (e, t) {
                return f.call(e, this, t);
              }
            : function (e) {
                return f.call(e, this);
              },
        );
    }
    r && hr(vr[o], "sham", !0);
  },
  $r = h,
  Lr = ee,
  Mr = J,
  Ur = L,
  Dr = function (e, t, n) {
    return t + (n ? Tr(e, t).length : 1);
  },
  Fr = function (e, t, n, r, o, i) {
    var c = n + e.length,
      a = r.length,
      u = Cr;
    return (
      void 0 !== o && ((o = _r(o)), (u = Ir)),
      Pr.call(i, u, function (i, u) {
        var l;
        switch (u.charAt(0)) {
          case "$":
            return "$";
          case "&":
            return e;
          case "`":
            return t.slice(0, n);
          case "'":
            return t.slice(c);
          case "<":
            l = o[u.slice(1, -1)];
            break;
          default:
            var f = +u;
            if (0 === f) return i;
            if (f > a) {
              var s = jr(f / 10);
              return 0 === s
                ? i
                : s <= a
                  ? void 0 === r[s - 1]
                    ? u.charAt(1)
                    : r[s - 1] + u.charAt(1)
                  : i;
            }
            l = r[f - 1];
        }
        return void 0 === l ? "" : l;
      })
    );
  },
  zr = function (e, t) {
    var n = e.exec;
    if ("function" == typeof n) {
      var r = n.call(e, t);
      if ("object" != typeof r)
        throw TypeError(
          "RegExp exec method returned something other than an Object or null",
        );
      return r;
    }
    if ("RegExp" !== Nr(e))
      throw TypeError("RegExp#exec called on incompatible receiver");
    return Ar.call(e, t);
  },
  Kr = Math.max,
  Br = Math.min;
kr("replace", 2, function (e, t, n, r) {
  var o = r.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,
    i = r.REPLACE_KEEPS_$0,
    c = o ? "$" : "$0";
  return [
    function (n, r) {
      var o = Ur(this),
        i = null == n ? void 0 : n[e];
      return void 0 !== i ? i.call(n, o, r) : t.call(String(o), n, r);
    },
    function (e, r) {
      if ((!o && i) || ("string" == typeof r && -1 === r.indexOf(c))) {
        var a = n(t, e, this, r);
        if (a.done) return a.value;
      }
      var u = $r(e),
        l = String(this),
        f = "function" == typeof r;
      f || (r = String(r));
      var s = u.global;
      if (s) {
        var p = u.unicode;
        u.lastIndex = 0;
      }
      for (var g = []; ; ) {
        var d = zr(u, l);
        if (null === d) break;
        if ((g.push(d), !s)) break;
        "" === String(d[0]) && (u.lastIndex = Dr(l, Lr(u.lastIndex), p));
      }
      for (var h, y = "", v = 0, x = 0; x < g.length; x++) {
        d = g[x];
        for (
          var b = String(d[0]),
            E = Kr(Br(Mr(d.index), l.length), 0),
            m = [],
            S = 1;
          S < d.length;
          S++
        )
          m.push(void 0 === (h = d[S]) ? h : String(h));
        var w = d.groups;
        if (f) {
          var O = [b].concat(m, E, l);
          void 0 !== w && O.push(w);
          var R = String(r.apply(void 0, O));
        } else R = Fr(b, l, E, m, w, r);
        E >= v && ((y += l.slice(v, E) + R), (v = E + b.length));
      }
      return y + l.slice(v);
    },
  ];
});
var Wr = {};
Wr[Ze("toStringTag")] = "z";
var Gr = "[object z]" === String(Wr),
  Vr = Gr,
  Yr = B,
  qr = Ze("toStringTag"),
  Xr =
    "Arguments" ==
    Yr(
      (function () {
        return arguments;
      })(),
    ),
  Hr = Vr
    ? Yr
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
                })((t = Object(e)), qr))
              ? n
              : Xr
                ? Yr(t)
                : "Object" == (r = Yr(t)) && "function" == typeof t.callee
                  ? "Arguments"
                  : r;
      },
  Jr = Gr
    ? {}.toString
    : function () {
        return "[object " + Hr(this) + "]";
      },
  Qr = Gr,
  Zr = ut.exports,
  eo = Jr;
Qr || Zr(Object.prototype, "toString", eo, { unsafe: !0 });
/*!
 * Handles finding a text string anywhere in the slides and showing the next occurrence to the user
 * by navigatating to that slide and highlighting it.
 *
 * @author Jon Snyder <snyder.jon@gmail.com>, February 2013
 */
export default function () {
  var e, t, n, r, o, i, c;
  function a() {
    (t = document.createElement("div")).classList.add("searchbox"),
      (t.style.position = "absolute"),
      (t.style.top = "10px"),
      (t.style.right = "10px"),
      (t.style.zIndex = 10),
      (t.innerHTML =
        '<input type="search" class="searchinput" placeholder="Search..." style="vertical-align: top;"/>\n\t\t</span>'),
      ((n = t.querySelector(".searchinput")).style.width = "240px"),
      (n.style.fontSize = "14px"),
      (n.style.padding = "4px 6px"),
      (n.style.color = "#000"),
      (n.style.background = "#fff"),
      (n.style.borderRadius = "2px"),
      (n.style.border = "0"),
      (n.style.outline = "0"),
      (n.style.boxShadow = "0 2px 18px rgba(0, 0, 0, 0.2)"),
      (n.style["-webkit-appearance"] = "none"),
      e.getRevealElement().appendChild(t),
      n.addEventListener(
        "keyup",
        function (t) {
          switch (t.keyCode) {
            case 13:
              t.preventDefault(),
                (function () {
                  if (i) {
                    var t = n.value;
                    "" === t
                      ? (c && c.remove(), (r = null))
                      : ((c = new f("slidecontent")),
                        (r = c.apply(t)),
                        (o = 0));
                  }
                  r &&
                    (r.length && r.length <= o && (o = 0),
                    r.length > o && (e.slide(r[o].h, r[o].v), o++));
                })(),
                (i = !1);
              break;
            default:
              i = !0;
          }
        },
        !1,
      ),
      l();
  }
  function u() {
    t || a(), (t.style.display = "inline"), n.focus(), n.select();
  }
  function l() {
    t || a(), (t.style.display = "none"), c && c.remove();
  }
  function f(t, n) {
    var r = document.getElementById(t) || document.body,
      o = n || "EM",
      i = new RegExp("^(?:" + o + "|SCRIPT|FORM)$"),
      c = ["#ff6", "#a0ffff", "#9f9", "#f99", "#f6f"],
      a = [],
      u = 0,
      l = "",
      f = [];
    (this.setRegex = function (e) {
      (e = e.replace(/^[^\w]+|[^\w]+$/g, "").replace(/[^\w'-]+/g, "|")),
        (l = new RegExp("(" + e + ")", "i"));
    }),
      (this.getRegex = function () {
        return l
          .toString()
          .replace(/^\/\\b\(|\)\\b\/i$/g, "")
          .replace(/\|/g, " ");
      }),
      (this.hiliteWords = function (t) {
        if (null != t && t && l && !i.test(t.nodeName)) {
          if (t.hasChildNodes())
            for (var n = 0; n < t.childNodes.length; n++)
              this.hiliteWords(t.childNodes[n]);
          var r, s;
          if (3 == t.nodeType)
            if ((r = t.nodeValue) && (s = l.exec(r))) {
              for (var p = t; null != p && "SECTION" != p.nodeName; )
                p = p.parentNode;
              var g = e.getIndices(p),
                d = f.length,
                h = !1;
              for (n = 0; n < d; n++)
                f[n].h === g.h && f[n].v === g.v && (h = !0);
              h || f.push(g),
                a[s[0].toLowerCase()] ||
                  (a[s[0].toLowerCase()] = c[u++ % c.length]);
              var y = document.createElement(o);
              y.appendChild(document.createTextNode(s[0])),
                (y.style.backgroundColor = a[s[0].toLowerCase()]),
                (y.style.fontStyle = "inherit"),
                (y.style.color = "#000");
              var v = t.splitText(s.index);
              (v.nodeValue = v.nodeValue.substring(s[0].length)),
                t.parentNode.insertBefore(y, v);
            }
        }
      }),
      (this.remove = function () {
        for (
          var e, t = document.getElementsByTagName(o);
          t.length && (e = t[0]);

        )
          e.parentNode.replaceChild(e.firstChild, e);
      }),
      (this.apply = function (e) {
        if (null != e && e)
          return this.remove(), this.setRegex(e), this.hiliteWords(r), f;
      });
  }
  return {
    id: "search",
    init: function (n) {
      (e = n).registerKeyboardShortcut("CTRL + Shift + F", "Search"),
        document.addEventListener(
          "keydown",
          function (e) {
            "F" == e.key &&
              (e.ctrlKey || e.metaKey) &&
              (e.preventDefault(),
              t || a(),
              "inline" !== t.style.display ? u() : l());
          },
          !1,
        );
    },
    open: u,
  };
}