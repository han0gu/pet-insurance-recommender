from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 제11조(손해방지의무)의 제1항 제2호의 제3자로부터 손해의 배상을<br>받을 수 있는 그 권리를 지키거나 행사하기 위하여 '
 "지출한 필요 또는 유익하였<br>던 비용</p><footer id='25' style='font-size:14px'>- 22 "
 "-</footer><h1 id='26' style='font-size:14px'>【설명】</h1><br><p id='27' "
 "data-category='paragraph' style='font-size:14px'>제3자로부터 손해의 배상을 받을 수 있는 경우에 "
 '피보험자가'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
