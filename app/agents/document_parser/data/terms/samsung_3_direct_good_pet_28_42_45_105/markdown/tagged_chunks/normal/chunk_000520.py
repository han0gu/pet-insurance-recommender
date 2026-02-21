from langchain_core.documents import Document

chunk = Document(
    page_content=('- 더합니다.\n'
 '- ⑥ 제4항에도 불구하고 동일한 질병에 대한 입원이라도 반려견 위탁비용이 지급된 최종\n'
 '- 입원의 퇴원일로부터 180일이 경과하여 개시한 입원은 새로운 입원으로 간주합니다.\n'
 '- 다만, 아래와 같이 반려견 위탁비용이 지급된 최종입원일부터 180일이 경과하도록 퇴\n'
 '- 원없이 계속 입원중인 경우에는 반려견 위탁비용이 지급된 최종 입원일의 그 다음날\n'
 '- 을 퇴원일로 봅니다.\n'
 '![image](/image/placeholder)\n'
 '- Chart Type: bar\n'
 '|  | 보상 | 보상제외 |\n'
 '| --- | --- | --- |'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000520',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
