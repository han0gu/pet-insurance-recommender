from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험기간 중 최초로 발견된 경우에는<br>해당 보험기간에 한하여 보상합니다.)<br>② 다음 정한 질병 및 이에 기인하는 '
 '질병(다만, 질병의<br>발생일로부터 과거 1년 이내의 예방접종 기록이 있는<br>경우에는 보상합니다.)<br>: 파보 바이러스 감염, '
 "디스템퍼 바이러스 감염, 파라</p><footer id='41' style='font-size:14px'>120</footer><p "
 "id='42' data-category='paragraph' style='font-size:18px'>인플루엔자 감염, 전염성 간염,"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000521',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
