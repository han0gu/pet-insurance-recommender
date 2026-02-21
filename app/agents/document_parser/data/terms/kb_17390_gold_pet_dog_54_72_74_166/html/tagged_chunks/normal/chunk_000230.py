from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 금융회사(우체국을 포함<br>합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으<br>로 '
 "대신합니다.</p><br><p id='41' data-category='paragraph' style='font-size:14px'>66 "
 "KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table id='42' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>용 어 풀 "
 '이'),
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
 'indexing': {'chunk_id': 'chunk_000230',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
