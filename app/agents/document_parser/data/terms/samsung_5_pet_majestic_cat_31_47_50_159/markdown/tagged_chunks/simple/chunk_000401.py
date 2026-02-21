from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관 일반사항 제7조(보험금을 지급하지 않는 사유)는 제외합니다.\n'
 '- ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 납입면제를 적용하지 않거나\n'
 '- 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보\n'
 '- 험료 납입면제) 및 제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다.\n'
 '- ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해약환급금)를 적용하지 않으며 보통\n'
 '- 약관 제36조(해약환급금)을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000401',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
