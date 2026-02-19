from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상\n'
 '파보바이러스 감염증 | 파보바이러스에 감염되어 구토와 설사 등의 증상을 일으킴\n'
 '디스템퍼바이러스 감염증 | 디스템퍼바이러스에 감염되어 호흡기 질환과 신경증상을 일으킴\n'
 '파라인플루엔자 감염증 | 파라인플루엔자에 감염되어, 기침, 가래, 콧물 등의 증상을 일으킴\n'
 '아데노바이러스 2형 감염증 | 아데노바이러스 2형 바이러스에 감염되어 호흡기 증상 등을 일으킴'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000365',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
