from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 피보험자가 제7조(손해방지의무) 제1항 제1호의 손해의 방지 또는 경감을 위하여 지출한 필 요 또는 유익하였던 비용 나. 피보험자가 '
 '제7조(손해방지의무) 제1항 제2호의 제3자로부터 손해의 배상을 받을 수 있는 그 권리를 지키거나 행사하기 위하여 지출한 필요 또는 '
 '유익하였던 비용 다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용 라. 보험증권상의 보상한도액내의 금액에 '
 '대한 공탁보증보험료. 그러나 회사는 그러한 보증을 제공할 책임은 부담하지 않습니다. 마'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
