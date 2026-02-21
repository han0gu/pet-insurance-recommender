from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관련된 상해 또는 질병 치료에 대한 비용\n'
 '- 14. 피부병(외이염, 면역성 피부병(아토피, 알러지 포함), 세균감염, 곰팡이감염, 기생충 감염, 호르\n'
 '- 몬성 피부병, 피부트러블을 포함) 치료에 대한 비용\n'
 '- 15. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료 및 이와 동종의 비용\n'
 '- 16. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다.\n'
 '파보바이러스 감염증, 디스템퍼바이러스 감염증, 파라인플루엔자 감염증, 전염성 간염, 아데노바이러스'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
