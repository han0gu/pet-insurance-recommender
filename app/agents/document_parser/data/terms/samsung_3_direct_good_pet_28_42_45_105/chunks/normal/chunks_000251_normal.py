from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[자동대출납입]\n'
 '보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대출납입을 신청하면 해당 보험 상품의 해약환급금 범위 내에서 납입할 보험료를 '
 '자동적으로 대출하여 이를 보험료 납입에 충당하는 서비스를 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
