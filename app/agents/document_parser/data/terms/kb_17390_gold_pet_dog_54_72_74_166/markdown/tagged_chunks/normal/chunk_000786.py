from langchain_core.documents import Document

chunk = Document(
    page_content=('번째 갱신계약 보험료표</td><td>6,300원</td><td>8,000원</td><td>9,900원</td><td>12,500원 '
 '…</td><td>나이증가 위험률상승</td><td>해 및 '
 '질</td></tr><tr><td>:</td><td>:</td><td>:</td><td>:</td><td>:</td><td>:</td><td>병</td></tr></tbody></table> '
 '|'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000786',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
