from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 대상이 되는 항목 | 수가코드 |\n'
 '| 체내고정용금속제거술 주 : 골에 삽입한 금속편이나 금속정 등을 간단히 제거한 경우 근막절개 하에 실시한 경우에는 770.08점을 산 '
 '정하고, 근막절개 없이 실시한 경우에는 501.67점을 산정한다. | N0978, N0979 |\n'
 '| 가. 골반골, 대퇴골 | N0972 |\n'
 '| 나. 상완골, 견갑골 | N0973 |\n'
 '| 다. 전완골, 하퇴골 |  |\n'
 '| (1)요골과 척골중하나, 경골과 비골중 하나 | N0977 |\n'
 '| (2)요척골 동시, 경비골 동시 | N0974 |'),
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
 'indexing': {'chunk_id': 'chunk_000975',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
