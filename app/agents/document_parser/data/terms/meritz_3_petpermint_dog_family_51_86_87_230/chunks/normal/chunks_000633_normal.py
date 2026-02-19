from langchain_core.documents import Document

chunk = Document(
    page_content=('㉠ 피보험자가「배상책임 관련 특별약관 일반조항」제8 조(손해방지의무)의 제1항 제1호의 손해의 방지 또 는 경감을 위하여 지출한 필요 '
 '또는 유익하였던 비 용 ㉡ 피보험자가「배상책임 관련 특별약관 일반조항」제8 조(손해방지의무)의 제1항 제2호의 제3자로부터 손 해의 '
 '배상을 받을 수 있는 그 권리를 지키거나 행사 하기 위하여 지출한 필요 또는 유익하였던 비용 ㉢ 피보험자가 지급한 소송비용, 변호사비용, '
 '중재, 화 해 또는 조정에 관한 비용 ㉣ 보험증권상 보상한도액내의 금액에 대한 공탁보증보 험료'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000633',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
