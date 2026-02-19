from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[계약자가 2명 이상인 경우]\n'
 '계약자가 2명 이상인 경우 계약 전 알릴 의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무 를 연대로 합니다.\n'
 '<용어풀이>\n'
 '[연대]\n'
 '2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되(지분만큼 분할하여 책임을 지는 것과 다름), 어느 1인의 '
 '이행으로 나머지 사람들도 책임을 면하게 되는 것을 말합니 다.\n'
 '제3관 계약자의 계약 전 알릴 의무 등\n'
 '제 15조 (계약 전 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 48},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
