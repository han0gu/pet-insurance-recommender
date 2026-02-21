from langchain_core.documents import Document

chunk = Document(
    page_content=('13. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. 다만, 질병의 발생일\n'
 '8. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으\n'
 '로부터 과거 1년 이내의 예방접종 기록이 있는 경우에는 보상합니다.\n'
 '로 이용함으로써 발생한 손해\n'
 '파보바이러스 감염증, 디스템퍼바이러스 감염증, 파라인플루엔자 감염증, 전염성 간염, 아\n'
 '9. 동물보호법 위반 등 동물학대에 기인하는 손해\n'
 '데노바이러스 2형 감염증, 코로나바이러스 감염증, 렙토스피라 감염증, 심상사상충 감염'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000404',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
