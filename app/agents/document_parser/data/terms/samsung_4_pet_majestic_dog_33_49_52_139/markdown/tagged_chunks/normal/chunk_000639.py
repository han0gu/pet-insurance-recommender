from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 계약자 또는 피보험자가 지출한 아래의 비용\n'
 '- 가. 피보험자가 제8조(손해방지의무) 제1항 제1호의 손해의 방지 또는 경감을 위하\n'
 '- 여 지출한 필요 또는 유익하였던 비용\n'
 '- 나. 피보험자가 제8조(손해방지의무) 제1항 제2호의 제3자로부터 손해의 배상을\n'
 '- 받을 수 있는 그 권리를 지키거나 행사하기 위하여 지출한 필요 또는 유익하\n'
 '- 였던 비용\n'
 '- 다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용\n'
 '- 라. 보험증권상 보상한도액 내의 금액에 대한 공탁보증보험료. 그러나 회사는 그러'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000639',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
