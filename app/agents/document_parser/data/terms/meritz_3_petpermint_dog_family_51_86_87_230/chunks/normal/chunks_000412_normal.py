from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험기간 중 최초로 발견된 경우에는 해당 보험기간에 한하여 보상합니다.) ② 다음 정한 질병 및 이에 기인하는 질병(다만, '
 '질병의 발생일로부터 과거 1년 이내의 예방접종 기록이 있는 경우에는 보상합니다.) : 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라 '
 '인플루엔자 감염, 전염성 간염, 아데노 바이러스 2 형 감염, 광견병, 코로나 바이러스 감염, 렙토스피 라 감염, 필라리아(심장사상충) '
 '감염, 인플루엔자 감염 ③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료 ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000412',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
