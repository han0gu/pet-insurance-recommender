from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 뼈와 관절의 영역\n'
 'Wobbler증후군, 팔꿈치 관절형성부전, 팔꿈치 관절 척골 이탈, 팔꿈치 관절요골 이탈, 앞발 허리골의 만곡증, 대퇴골두 '
 '괴사증(Legg-calv-perthes disease)\n'
 '나. 눈과 구강치\n'
 '눈구멍 형성부전, 눈꺼풀 외번, 눈꺼풀 내번, 망막 변성의 진행, 하악골의 염증성 질환, 이 및 턱의 형성부전\n'
 '다. 하기와 같은 선천성 결손\n'
 '선천성 난청, Achalasia(식도·직장 등의 이완 불능증), 구개열, 동맥관 개존증'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'other',
            'risk_domains': ['joint', 'eye', 'dental', 'other']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
