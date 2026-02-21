from langchain_core.documents import Document

chunk = Document(
    page_content=('# 4) 뇌전증- 가) “뇌전증”이라 함은 돌발적 뇌파이상을 나타내는\n'
 '- 뇌질환으로 발작(경련, 의식장해 등)을 반복하는\n'
 '- 것을 말한다.\n'
 '- 나) 뇌전증 발작의 빈도 및 양상은 지속적인 항뇌전\n'
 '- 증제(항경련제) 약물로도 조절되지 않는 뇌전증\n'
 '- 을 말하며, 진료기록에 기재되어 객관적으로 확\n'
 '- 인되는 뇌전증 발작의 빈도 및 양상을 기준으로\n'
 '203# 한다.- 다) “심한 뇌전증 발작”이라 함은 월 8회 이상의 중\n'
 '- 증발작이 연 6개월 이상의 기간에 걸쳐 발생하\n'
 '- 고, 발작할 때 유발된 호흡장애, 흡인성 폐렴,'),
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
