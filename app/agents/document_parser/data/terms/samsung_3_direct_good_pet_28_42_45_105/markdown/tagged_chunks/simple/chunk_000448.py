from langchain_core.documents import Document

chunk = Document(
    page_content=('골 이탈, 앞발 허리골의 만곡증, 대퇴골두 괴사증(Legg-calv-perthes\n'
 'disease)\n'
 '나. 눈과 구강치\n'
 '눈구멍 형성부전, 눈꺼풀 외번, 눈꺼풀 내번, 망막 변성의 진행, 하악골의 염증\n'
 '성 질환, 이 및 턱의 형성부전\n'
 '다. 하기와 같은 선천성 결손\n'
 '선천성 난청, Achalasia(식도·직장 등의 이완 불능증), 구개열, 동맥관 개존증| <용어풀이> | <용어풀이> |\n'
 '| --- | --- |\n'
 '| 배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상 |'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000448',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
