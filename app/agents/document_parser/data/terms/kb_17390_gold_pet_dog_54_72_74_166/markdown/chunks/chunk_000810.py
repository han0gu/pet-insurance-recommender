from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상 약\n'
 '전환대상계약을 소득세법 제59조의4(특별세액공제) 성특도KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 137- 137 -| 계약 '
 '보험료는 | 보험료 납입영수증에 장애인전용 보장성보험료로 표시됩니다. |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
