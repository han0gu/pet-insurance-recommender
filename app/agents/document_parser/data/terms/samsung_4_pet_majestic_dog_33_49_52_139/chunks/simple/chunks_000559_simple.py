from langchain_core.documents import Document

chunk = Document(
    page_content=('눈구멍 형성부전, 눈꺼풀 외번, 눈꺼풀 내번, 망막 변성의 진행, 하악골의 염증 성 질환, 이 및 턱의 형성부전 다. 하기와 같은 선천성 '
 '결손 선천성 난청, Achalasia(식도 · 직장 등의 이완 불능증), 구개열, 동맥관 개존증\n'
 '<용어풀이>\n'
 '배꼽허니아 | 복부 내장의 탈장등으로 인해배꼽주변이 부풀어 오르는 증상\n'
 '파보바이러스 감염증 | 파보바이러스에 감염되어 구토와 설사등의증상을 일으킴\n'
 '디스템퍼바이러스 감염증 | 디스템퍼바이러스에 감염되어 호흡기 질환과 신경증상을 일으킴'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['eye', 'dental', 'joint', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000559',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
