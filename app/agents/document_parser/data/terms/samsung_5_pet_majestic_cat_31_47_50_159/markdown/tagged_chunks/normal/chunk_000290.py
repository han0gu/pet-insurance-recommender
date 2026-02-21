from langchain_core.documents import Document

chunk = Document(
    page_content=('확인할 수 있습니다.1. 상해 관련 특별약관1-1. 상해 후유장해(3~100%) 특별약관# 제1관 일반사항- ① 제2관 개별사항에서 '
 '정하지 않은 사항은 특별약관의 일반사항을 적용합니다.\n'
 '- ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 납입면제를 적용하지 않거나\n'
 '- 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보\n'
 '- 험료 납입면제) 및 제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000290',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
