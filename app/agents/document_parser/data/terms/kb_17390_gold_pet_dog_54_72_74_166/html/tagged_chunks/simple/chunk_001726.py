from langchain_core.documents import Document

chunk = Document(
    page_content=('중 다음의수가코드에 해당하는 의료행위를</td><td>말합니다.</td></tr></thead><tbody><tr><td>대상이 되는 '
 '항목</td><td>수가코드</td></tr><tr><td>체내고정용금속제거술 주 : 골에 삽입한 금속편이나 금속정 등을 간단히 제거한 '
 '경우 근막절개 하에 실시한 경우에는 770.08점을 산 정하고, 근막절개 없이 실시한 경우에는 501.67점을 '
 '산정한다.</td><td>N0978, '
 'N0979</td></tr><tr><td>가.</td><td></td></tr><tr><td>골반골, 대퇴골 나'),
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
 'indexing': {'chunk_id': 'chunk_001726',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
