from langchain_core.documents import Document

chunk = Document(
    page_content=('53 / 181\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일 단, 계약해당일 2월 29일이 없을 경우에는 2월 28일을 '
 '계약해당일로 합니다.\n'
 '제 25조 (특별약관의 소멸)\n'
 '각 특별약관의 보장을 따릅니다.\n'
 '제5관 보험료의 납입\n'
 '제 26조 (제1회 보험료 및 회사의 보장개시)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
