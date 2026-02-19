from langchain_core.documents import Document

chunk = Document(
    page_content=('82 / 181\n'
 '13. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. (다만, 질병의 발생일 로부터 과거 1년 이내의 예방접종 기록이 있는 '
 '경우에는 보상합니다.)\n'
 '파보바이러스 감염증, 디스템퍼바이러스 감염증, 파라인플루엔자 감염증, 전염성 간염, 아 데노바이러스 2형 감염증, 코로나바이러스 감염증, '
 '렙토스피라 감염증, 심상사상충 감염 증, 광견병, 켄넬코프\n'
 '14. 기관협착, 누루관시술과 관련된 상해 또는 질병 치료에 대한 비용 15. 아래의 유전적 또는 발달이상을 원인으로 하는 경우는 '
 '보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 83},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000515',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
