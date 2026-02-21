from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="2">고급형Ⅱ</td><td>입원</td><td>1일당 30만원 한도 1일당</td><td>300만원 '
 '한도</td><td>2,000만원 려동</td></tr><tr><td>통원</td><td>1일당 30만원 한도 '
 "1일당</td><td>300만원 한도</td><td>2,000만원 물</td></tr></tbody></table><p id='118' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01) 109</p><br><p'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000941',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
