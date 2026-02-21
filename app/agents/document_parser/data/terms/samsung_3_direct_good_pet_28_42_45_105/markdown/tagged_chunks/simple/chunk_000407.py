from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약ㆍ예방 접종비용 및 정기검\n'
 '나. 눈과 구강치\n'
 '진, 예방적 검사를 위한 비용\n'
 '눈구멍 형성부전, 눈꺼풀 외번, 눈꺼풀 내번, 망막 변성의 진행, 하악골의 염증\n'
 '2. 임신, 출산(제왕절개를 포함합니다), 인공유산과 관련된 비용 및 출산 후 증상 치료\n'
 '성 질환, 이 및 턱의 형성부전\n'
 '비용\n'
 '다. 하기와 같은 선천성 결손\n'
 '3. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용'),
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
 'indexing': {'chunk_id': 'chunk_000407',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
