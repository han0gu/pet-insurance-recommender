from langchain_core.documents import Document

chunk = Document(
    page_content=("id='270' style='font-size:20px'>12.</h1><br><p id='271' "
 "data-category='paragraph' style='font-size:14px'>별<br>제1조(보험금의 "
 '지급사유)<br>약<br>\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 '
 '"골절진단<br>관<br>(치아파절제외)"로 진단확정 되고 그 치료를 목적으로 체내에 삽입한 철심을 제<br>거하는 "골절철심제거술"을 '
 '받은 경우 이 특별약관의 보험가입금액을 연간 1회에<br>한하여 보험수익자에게'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000527',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
