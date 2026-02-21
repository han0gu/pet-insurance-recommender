from langchain_core.documents import Document

chunk = Document(
    page_content=('침범한 골절</td><td>T02</td></tr><tr><td>척추의 상세불명 부위의 '
 '골절</td><td>T08</td></tr><tr><td>팔의 상세불명 부위의 '
 '골절</td><td>T10</td></tr><tr><td>다리의 상세불명 부위의 '
 '골절</td><td>T12</td></tr><tr><td>상세불명의 신체부위의 '
 "골절</td><td>T14.2</td></tr></tbody></table><br><h1 id='51' "
 "style='font-size:14px'>주) 1"),
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
 'indexing': {'chunk_id': 'chunk_001721',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
