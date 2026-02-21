from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만 아래에 기재된 개(犬)는 이 보험의 가입 대상이 아닙니다. 1. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 '
 '초과하는 개(犬) 2. 판매점, 브리더 등이 매매(賣買)를 목적으로 사육․관리 하 는 개(犬) 3. 경찰견, 구조견, 군견, 사냥개 등 '
 '특수한 목적의 개(犬) (단, 맹도견, 청도견 등 장애인 안내견은 제외) 4. 투견, 경주견 등 흥행을 목적으로 사육․관리 하는 '
 "개(犬)</td></tr></tbody></table><br><h1 id='113' style='font-size:14px'>5"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000759',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
