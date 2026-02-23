from langchain_core.documents import Document

chunk = Document(
    page_content=('- 20° 이상의 척추측만증(척추가 옆으로 휘어지는\n'
 '- 증상) 변형이 있을 때\n'
 '- 나) 척추체(척추뼈 몸통) 한 개의 압박률이 60%이상\n'
 '- 인 경우 또는 한 운동단위 내에 두 개 이상 척추\n'
 '- 체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈\n'
 '- 몸통)의 압박률의 합이 90% 이상일 때\n'
 '10) 뚜렷한 기형이란 다음 중 어느 하나에 해당하는 경\n'
 '우를 말한다.- 가) 척추(등뼈)의 골절 또는 탈구 등으로 15° 이상\n'
 '- 의 척추전만증(척추가 앞으로 휘어지는 증상),\n'
 '- 척추후만증(척추가 뒤로 휘어지는 증상) 또는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
