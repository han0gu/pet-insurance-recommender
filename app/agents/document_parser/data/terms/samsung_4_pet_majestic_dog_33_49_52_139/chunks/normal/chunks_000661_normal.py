from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 뼈와 관절의 영역\n'
 'Wobbler증후군, 팔꿈치 관절형성부전, 팔꿈치 관절 척골 이탈, 팔꿈치 관절요 골 이탈, 앞발 허리골의 만곡증, 대퇴골두 '
 '괴사증(Legg-calv-perthes disease) 나. 눈과 구강치 눈구멍 형성부전, 눈꺼풀 외번, 눈꺼풀 내번, 망막 변성의 '
 '진행, 하악골의 염증 성 질환, 이 및 턱의 형성부전 다. 하기와 같은 선천성 결손 선천성 난청, Achalasia(식도·직장 등의 이완 '
 '불능증), 구개열, 동맥관 개존증\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['joint', 'eye', 'dental', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000661',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
