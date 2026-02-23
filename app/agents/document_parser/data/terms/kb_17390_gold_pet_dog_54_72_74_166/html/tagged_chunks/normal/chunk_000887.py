from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의<br>활용을 중단할 것을 요구하는 경우, 회사는 '
 '전화(음성녹음) 방법으로 전환하<br>여 제1항에 따른 납입최고(독촉) 등을 실시할 것<br>4. 전자적 상품설명장치에 안내의 속도와 '
 "음량을 조절할 수 있는 기능을 갖출 것</p><br><p id='62' data-category='paragraph' "
 "style='font-size:16px'>5"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000887',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
