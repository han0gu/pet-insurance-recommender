from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 4. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 '
 '임신ㆍ출산과 관련 된 질병 치료에 대한 비용 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환, 배꼽허니아(배꼽부위탈장), '
 '항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 손톱깎기 등의 처치비용 6. 미용으로 인한 비용 7. 귀 성형, 꼬리 '
 '성형, 성대 제거 및 미용성형 등 질병치료가 아닌 수술에 소요되는 비용 8'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000359',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
