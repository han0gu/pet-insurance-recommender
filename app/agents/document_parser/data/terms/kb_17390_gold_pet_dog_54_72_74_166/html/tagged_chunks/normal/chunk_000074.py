from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table id='90' "
 "style='font-size:20px'><thead></thead><tbody><tr><td></td><td><table><thead></thead><tbody><tr><td></td></tr><tr><td>예 "
 '시 보험금을 나누어 지급받을 경우</td></tr></tbody></table></td></tr><tr><td>예) 보험금: 6천만원, '
 '보험금</td><td>지급일자: 2024년 4월 10일 일때 보험금을 일시'),
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
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
