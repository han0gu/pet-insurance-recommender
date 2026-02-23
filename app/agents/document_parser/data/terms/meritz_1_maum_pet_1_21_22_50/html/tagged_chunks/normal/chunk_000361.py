from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 이 조에서 같다)가 해당 과세<br>기간에 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보험의 보험계약에<br>따라 '
 '지급하는 다음 각 호의 보험료를 지급한 경우 그 금액의 100분의 12(제1호의<br>경우에는 100분의 15)에 해당하는 금액을 해당 '
 '과세기간의 종합소득산출세액에서 공<br>제한다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000361',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
