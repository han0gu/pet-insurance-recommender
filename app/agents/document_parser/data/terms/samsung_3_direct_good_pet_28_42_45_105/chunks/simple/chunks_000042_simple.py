from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[연대]\n'
 '2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되(지분만큼 분할하여 책임을 지는 것과 다름), 어느 1인의 '
 '이행으로 나머지 사람들도 책임을 면하게 되는 것을 말합니 다.\n'
 '제3관 계약자의 계약 전 알릴 의무 등\n'
 '제13조 (계약 전 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 32},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
