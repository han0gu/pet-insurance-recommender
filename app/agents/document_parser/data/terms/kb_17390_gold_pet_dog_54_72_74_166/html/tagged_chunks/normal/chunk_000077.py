from langchain_core.documents import Document

chunk = Document(
    page_content=('. (금융감독원 홈페이지(www.fss.or.kr)의 '
 '"업무자료-보</td></tr></tbody></table></td></tr></tbody></table><br><h1 id=\'91\' '
 'style=\'font-size:14px\'>험상품자료"에서 확인할 수 있습니다)</h1><p id=\'92\' '
 "data-category='paragraph' style='font-size:14px'>제11조(주소변경통지)<br>\uf000 "
 '계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는<br>연락처가 변경된 경우에는 지체없이 그'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000077',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
