from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(단체요율의 적용)\n'
 '① 제1조(계약의 적용 범위)에 해당하는 단체는 단체요율을 적용할 수 있습니다. 다만, 제3 종 단체는 구성원이 명확하고 위험의 동질성이 '
 '확보되어야 합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 101,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
