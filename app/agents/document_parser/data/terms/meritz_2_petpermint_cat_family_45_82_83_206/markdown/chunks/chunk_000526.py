from langchain_core.documents import Document

chunk = Document(
    page_content=('# 나. 장해판정기준- 1) 청력장해는 순음청력검사 결과에 따라 데시벨(dB :\n'
 '- decibel)로서 표시하고 3회 이상 청력검사를 실시한\n'
 '- 후 적용한다. 다만, 각 측정치의 결과값 차이가\n'
 '- ±10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해\n'
 '- 객관적인 장해 상태를 재평가 하여야 한다.\n'
 '- 2) “한 귀의 청력을 완전히 잃었을 때”라 함은 순음청력\n'
 '- 검사 결과 평균순음역치가 90dB이상인 경우를 말한다.\n'
 '- 3) “심한 장해를 남긴 때”라 함은 순음청력검사 결과\n'
 '- 평균순음역치가 80dB이상인 경우에 해당되어, 귀에다'),
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
