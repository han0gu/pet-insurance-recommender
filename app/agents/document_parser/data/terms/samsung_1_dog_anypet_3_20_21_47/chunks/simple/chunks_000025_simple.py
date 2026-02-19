from langchain_core.documents import Document

chunk = Document(
    page_content=('선천성 난청, Achalasia(식도·직장 등의 이완 불능증), 구개열, 동맥관 개존증\n'
 '【배꼽허니아】 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상 【파보바이러스 감염증】 파보바이러스에 감염되어 구토와 설사 '
 '등의 증상을 일으킴 【디스템퍼바이러스 감염증】 디스템퍼바이러스에 감염되어 호흡기 질환과 신경증상을 일으킴 【파라인플루엔자 감염증】 '
 '파라인플루엔자에 감염되어, 기침, 가래, 콧물 등의 증상을 일으킴 【아데노바이러스 2형 감염증】 아데노바이러스 2형 바이러스에 감염되어 '
 '호흡기 증상 등을 일으 킴'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['other', 'digestive', 'skin', 'other', 'other']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
