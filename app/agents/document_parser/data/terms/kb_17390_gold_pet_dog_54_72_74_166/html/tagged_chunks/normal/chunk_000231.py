from langchain_core.documents import Document

chunk = Document(
    page_content=("어 풀 이 납입기일</td></tr></tbody></table><br><h1 id='43' "
 "style='font-size:14px'>계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.</h1><p id='44' "
 "data-category='list' style='font-size:14px'>제27조(보험료의 자동대출납입)<br>\uf000 계약자는 "
 '제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)<br>에 따른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 '
 '정한 방법에 따라<br>보험료의 자동대출납입을'),
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
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
