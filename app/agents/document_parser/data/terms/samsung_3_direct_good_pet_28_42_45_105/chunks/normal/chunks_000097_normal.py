from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자에게 지급하고, 이 계약은 더 이상 효력이 없습니다.\n'
 '<용어풀이>\n'
 '[계약자적립액]\n'
 '장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 중 일정액을 회사가 적립해 둔 금액을 말합니다.\n'
 '제5관 보험료의 납입\n'
 '제24조 (제1회 보험료 및 회사의 보장개시)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
