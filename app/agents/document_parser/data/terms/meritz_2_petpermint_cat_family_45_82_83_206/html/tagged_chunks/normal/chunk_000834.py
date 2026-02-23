from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 정한 회사가 보험금을 지급하지 않는<br>기간 중에 회사가 지정한 질병(이하「특정질병」이라 '
 '합니<br>다)(【별첨(특정질병 분류표(반려묘))】)을 직접적인 원인<br>으로 계약에서 정한 보험금 지급사유가 발생한 경우에는 '
 "회<br>사는 보험금을 지급하지 않습니다.</p><br><p id='87' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부<br>담보 기간」이라 "
 '합니다)은 특정질병의 상태에 따라「1개월<br>부터'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000834',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
