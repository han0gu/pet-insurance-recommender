from langchain_core.documents import Document

chunk = Document(
    page_content=('지의 후유장해에 대한 후유장해보험금이 지급된 것으로 보\n'
 '고 최종 후유장해 상태에 해당되는 후유장해보험금에서 이\n'
 '를 차감하여 지급합니다.# 제5조(보험금을 지급하지 않는 사유)\uf000 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생\n'
 '한 때에는 보험금을 지급하지 않습니다.① 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자\n'
 '가 심신상실 등으로 자유로운 의사결정을 할 수 없는\n'
 '상태에서 자신을 해친 경우에는 보험금을 지급합니다.# 【심신상실】정신병, 정신박약, 심한 의식장애 등의 심신장애로 인하'),
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
