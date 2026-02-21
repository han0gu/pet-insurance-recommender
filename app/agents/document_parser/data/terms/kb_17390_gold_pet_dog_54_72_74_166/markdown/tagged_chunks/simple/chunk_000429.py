from langchain_core.documents import Document

chunk = Document(
    page_content=('동일한 질병에 대한 입원이라도 질병입원일당이 지급된 최종 입원의 퇴원일로부\n'
 '터 180일이 지나서 개시한 입원은 새로운 입원으로 봅니다. 다만, 아래와 같이- 94 -| 질병입원일당이 중인 경우에는 | 지급된 '
 '최종입원일부터 180일이 경과하도록 퇴원없이 계속 입원 입원일당이 지급된 최종입원일의 그 다음날을 |\n'
 '| --- | --- |\n'
 '|  | 퇴원일로 봅니다. ![image](/image/placeholder)\n'
 '예 시\n'
 '입원일당이 지급된\n'
 '최초입원일 보장재개\n'
 '최종입원일\n'
 '퇴원없이\n'
 '…\n'
 '계속 입원\n'
 '보장 보장제외 보장\n'
 '… |'),
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
 'indexing': {'chunk_id': 'chunk_000429',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
