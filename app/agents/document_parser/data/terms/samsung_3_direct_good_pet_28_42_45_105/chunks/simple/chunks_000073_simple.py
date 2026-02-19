from langchain_core.documents import Document

chunk = Document(
    page_content=('제18조 (청약의 철회)\n'
 '① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건강상태 진단을 지원하는 계약, '
 '보험기간이 90일 이내인 계약 또는 전문금융 a , " 지기 해군 JUOL BIOL 二 ▲ ) 人 · ···'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
