from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제1조(계약의 적용 범위)에 해당하는 단체가 피보험자를 확정할 수 있고 계약의 일괄적 관 리가 가능하며, 규약에 따라 계약을 체결하는 '
 '경우 피보험자의 서면에 의한 동의를 얻지 않아도 되며, 계약자에게만 보험증권을 드릴 수 있습니다. ② 제1항의 규약은 보험의 종류 및 '
 '일괄 가입에 관한 사항이 포함되어야 하며, 동의 또는 협의를 통하여 피보험자들의 의사가 규약에 반영될 수 있어야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
