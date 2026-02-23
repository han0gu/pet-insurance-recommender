from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의 활용<br>을 중단할 것을 요구하는 경우, 회사는 전화 '
 "(음성녹음) 방법으로 전환하여 제1항에<br>따른 납입최고(독촉) 등을 실시할 것</p><br><p id='58' "
 "data-category='list' style='font-size:14px'>4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 "
 '있는 기능을 갖출 것<br>5'),
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
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
